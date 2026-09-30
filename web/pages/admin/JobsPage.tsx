/**
 * Admin > Jobs: every user's training runs and exports. An admin can stop a queued or running run, and give a
 * failed export a fresh set of attempts. Both act through the same guarded transitions the owner's own
 * controls use, so a row that has just moved on answers with an error instead of being corrupted.
 */

import { Badge, Button, Group, Pagination, Select, Tabs, Text, TextInput } from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { ArrowClockwiseIcon, MagnifyingGlassIcon, StopIcon } from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { DataTable, type DataTableColumn, PageHeader, QueryBoundary, StatusBadge } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  adminCancelRun,
  adminRequeueExport,
  getAdminListExportsQueryOptions,
  getAdminListRunsQueryOptions,
} from '@public/lib/api/generated/admin/admin'
import type { AdminExportOut, AdminRunOut } from '@public/lib/api/generated/models'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

const PAGE_SIZE = 25

const RUN_STATUSES = ['queued', 'running', 'succeeded', 'failed', 'canceled']
const EXPORT_STATUSES = ['pending', 'converting', 'assembling', 'ready', 'failed']
const EXPORT_COLORS: Record<string, string> = {
  pending: 'gray',
  converting: 'cyan',
  assembling: 'cyan',
  ready: 'teal',
  failed: 'red',
}

const when = (value: string | null | undefined) => (value ? new Date(value).toLocaleString() : '—')

function Owner({ email }: { email: string }) {
  return (
    <Text size="sm" c="dimmed">
      {email}
    </Text>
  )
}

/** Status filter, owner search and paging, shared by both tables. */
function useJobFilters() {
  const [status, setStatus] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [user] = useDebouncedValue(search.trim(), 250)
  return { status, setStatus, search, setSearch, page, setPage, user }
}

function Toolbar({ statuses, filters }: { statuses: string[]; filters: ReturnType<typeof useJobFilters> }) {
  return (
    <Group gap="xs">
      <TextInput
        size="xs"
        placeholder="Owner email"
        leftSection={<MagnifyingGlassIcon size={14} />}
        value={filters.search}
        onChange={(e) => {
          filters.setSearch(e.currentTarget.value)
          filters.setPage(1)
        }}
        w={220}
      />
      <Select
        size="xs"
        placeholder="Any status"
        clearable
        data={statuses}
        value={filters.status}
        onChange={(v) => {
          filters.setStatus(v)
          filters.setPage(1)
        }}
        w={150}
      />
    </Group>
  )
}

function RunsTable() {
  const queryClient = useQueryClient()
  const filters = useJobFilters()
  const params = {
    status: filters.status ?? undefined,
    user: filters.user || undefined,
    page: filters.page,
    page_size: PAGE_SIZE,
  }
  const { data, isLoading, isError, refetch } = useQuery({
    ...getAdminListRunsQueryOptions(params),
    placeholderData: (previous) => previous,
    refetchInterval: 5000,
  })

  const cancel = useMutation({
    mutationFn: (runId: string) => adminCancelRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: getAdminListRunsQueryOptions().queryKey.slice(0, 1) })
      notifications.show({ title: 'Cancel requested', message: 'The run will stop shortly.', color: 'gray' })
    },
    onError: (error) => notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' }),
  })

  const confirmCancel = (run: AdminRunOut) =>
    modals.openConfirmModal({
      title: 'Cancel training run',
      children: (
        <Text size="sm">
          Stop “{run.name}” ({run.ownerEmail})? A run that is training stops at its next epoch.
        </Text>
      ),
      labels: { confirm: 'Cancel run', cancel: 'Keep running' },
      confirmProps: { color: 'red' },
      onConfirm: () => cancel.mutate(run.id),
    })

  const columns: DataTableColumn<AdminRunOut>[] = [
    {
      key: 'run',
      header: 'Run',
      render: (r) => (
        <div>
          <Text size="sm" fw={600}>
            {r.name}
          </Text>
          <Text size="xs" c="dimmed">
            {r.projectName} · {r.task} · {r.backend}
          </Text>
          {r.status === 'failed' && r.failedMessage && (
            <Text size="xs" c="red" lineClamp={2} title={r.failedMessage} maw={360}>
              {r.failedMessage}
            </Text>
          )}
        </div>
      ),
    },
    { key: 'owner', header: 'Owner', render: (r) => <Owner email={r.ownerEmail} /> },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (r) => <StatusBadge value={r.status} colorMap={STATUS_COLORS} />,
    },
    {
      key: 'created',
      header: 'Created',
      fit: true,
      render: (r) => (
        <Text size="sm" c="dimmed">
          {when(r.createdAt)}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (r) =>
        (r.status === 'queued' || r.status === 'running') && (
          <Button
            size="xs"
            variant="subtle"
            color="red"
            leftSection={<StopIcon size={14} />}
            onClick={() => confirmCancel(r)}
          >
            Cancel
          </Button>
        ),
    },
  ]

  const totalPages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE))
  return (
    <div className="flex flex-col gap-2">
      <Toolbar statuses={RUN_STATUSES} filters={filters} />
      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        <DataTable columns={columns} data={data?.runs ?? []} getRowKey={(r) => r.id} emptyMessage="No matching runs" />
        {totalPages > 1 && <Pagination value={filters.page} onChange={filters.setPage} total={totalPages} size="sm" />}
      </QueryBoundary>
    </div>
  )
}

function ExportsTable() {
  const queryClient = useQueryClient()
  const filters = useJobFilters()
  const params = {
    status: filters.status ?? undefined,
    user: filters.user || undefined,
    page: filters.page,
    page_size: PAGE_SIZE,
  }
  const { data, isLoading, isError, refetch } = useQuery({
    ...getAdminListExportsQueryOptions(params),
    placeholderData: (previous) => previous,
    refetchInterval: 5000,
  })

  const requeue = useMutation({
    mutationFn: (exportId: string) => adminRequeueExport(exportId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: getAdminListExportsQueryOptions().queryKey.slice(0, 1) })
      notifications.show({ title: 'Requeued', message: 'The export will be tried again.', color: 'teal' })
    },
    onError: (error) => notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' }),
  })

  const columns: DataTableColumn<AdminExportOut>[] = [
    {
      key: 'export',
      header: 'Export',
      render: (e) => (
        <div>
          <Text size="sm" fw={600}>
            {e.format}
          </Text>
          <Text size="xs" c="dimmed">
            {e.runName}
          </Text>
          {e.status === 'failed' && (e.failedMessage ?? e.lastError) && (
            <Text size="xs" c="red" lineClamp={2} title={e.failedMessage ?? e.lastError ?? undefined} maw={360}>
              {e.failedMessage ?? e.lastError}
            </Text>
          )}
        </div>
      ),
    },
    { key: 'owner', header: 'Owner', render: (e) => <Owner email={e.ownerEmail} /> },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (e) => (
        <Group gap={6} wrap="nowrap">
          <StatusBadge value={e.status} colorMap={EXPORT_COLORS} />
          {e.attempt > 0 && (
            <Badge size="xs" variant="outline" color="gray">
              {e.attempt}/{e.maxAttempts}
            </Badge>
          )}
        </Group>
      ),
    },
    {
      key: 'created',
      header: 'Created',
      fit: true,
      render: (e) => (
        <Text size="sm" c="dimmed">
          {when(e.createdAt)}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (e) =>
        e.status === 'failed' && (
          <Button
            size="xs"
            variant="subtle"
            leftSection={<ArrowClockwiseIcon size={14} />}
            loading={requeue.isPending && requeue.variables === e.id}
            onClick={() => requeue.mutate(e.id)}
          >
            Requeue
          </Button>
        ),
    },
  ]

  const totalPages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE))
  return (
    <div className="flex flex-col gap-2">
      <Toolbar statuses={EXPORT_STATUSES} filters={filters} />
      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        <DataTable
          columns={columns}
          data={data?.exports ?? []}
          getRowKey={(e) => e.id}
          emptyMessage="No matching exports"
        />
        {totalPages > 1 && <Pagination value={filters.page} onChange={filters.setPage} total={totalPages} size="sm" />}
      </QueryBoundary>
    </div>
  )
}

export function JobsPage() {
  const [tab, setTab] = useState<string>('runs')
  return (
    <div className="flex flex-col gap-2">
      <PageHeader title="Jobs" description="Every user's training runs and exports." />
      <Tabs value={tab} onChange={(v) => v && setTab(v)}>
        <Tabs.List>
          <Tabs.Tab value="runs">Training runs</Tabs.Tab>
          <Tabs.Tab value="exports">Exports</Tabs.Tab>
        </Tabs.List>
      </Tabs>
      {tab === 'runs' ? <RunsTable /> : <ExportsTable />}
    </div>
  )
}

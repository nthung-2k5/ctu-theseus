/**
 * Admin > Workers: the remote GPU machines that train runs for this server. A worker is registered here, gets a token
 * shown exactly once, and from then on pulls work by itself (`python -m omnitrain.worker`). Nothing here starts a
 * machine: it only says which ones may take runs, whether they are there, and what they are doing.
 *
 * Which runs go to a worker is not chosen anywhere. The server hands a new run to an idle online worker that can
 * train its backend, and trains it itself if there is none (or the worker does not pick it up within seconds).
 */

import { Badge, Button, Code, Group, Modal, Stack, Text, TextInput } from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { PlusIcon, PowerIcon, TrashIcon } from '@phosphor-icons/react'
import {
  CopyField,
  DataTable,
  type DataTableColumn,
  PageHeader,
  QueryBoundary,
  StatusBadge,
} from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  adminCreateWorker,
  adminRevokeWorker,
  adminUpdateWorker,
  getAdminListWorkersQueryKey,
  getAdminListWorkersQueryOptions,
} from '@public/lib/api/generated/admin/admin'
import type { CreatedWorker, WorkerOut } from '@public/lib/api/generated/models'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

const STATE_COLORS: Record<string, string> = {
  training: 'blue',
  online: 'teal',
  offline: 'gray',
  disabled: 'yellow',
  revoked: 'red',
}

const when = (value: string | null | undefined) => (value ? new Date(value).toLocaleString() : 'never')

/** What a worker is doing right now, as one word. A switched-off worker that is still training reads as training. */
function stateOf(w: WorkerOut): string {
  if (w.revokedAt) return 'revoked'
  if (w.currentRunId) return 'training'
  if (!w.enabled) return 'disabled'
  return w.online ? 'online' : 'offline'
}

/** The command a new worker is started with, with the real token, and what it needs besides. */
function StartCommand({ token }: { token: string }) {
  const command = [
    `OMNITRAIN_API_URL=${window.location.origin}`,
    `OMNITRAIN_WORKER_TOKEN=${token}`,
    'S3_ENDPOINT=<object store address, as seen from the worker>',
    'S3_ACCESS_KEY=<key> S3_SECRET_KEY=<secret>',
    'python -m omnitrain.worker',
  ].join(' \\\n  ')
  return (
    <Code block fz="xs">
      {command}
    </Code>
  )
}

function TokenModal({ created, onClose }: { created: CreatedWorker | null; onClose: () => void }) {
  return (
    <Modal opened={created !== null} onClose={onClose} title="Worker registered" size="lg">
      {created && (
        <Stack gap="sm">
          <Text size="sm">
            Copy this token now: it is shown only once and cannot be recovered. If you lose it, revoke “
            {created.worker.name}” and register it again.
          </Text>
          <CopyField value={created.token} truncate={false} />
          <Text size="sm">
            Start the worker on the GPU machine. It must run the same commit of OmniTrain as this server, and reach the
            same object store buckets this server uses (the S3 values are the server's own). A prebuilt image is
            described in <Code>server/Dockerfile.worker</Code>.
          </Text>
          <StartCommand token={created.token} />
          <Group justify="flex-end">
            <Button onClick={onClose}>Done</Button>
          </Group>
        </Stack>
      )}
    </Modal>
  )
}

function RegisterModal({
  opened,
  onClose,
  onCreated,
}: {
  opened: boolean
  onClose: () => void
  onCreated: (created: CreatedWorker) => void
}) {
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const create = useMutation({
    mutationFn: (workerName: string) => adminCreateWorker({ name: workerName }),
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: getAdminListWorkersQueryKey() })
      setName('')
      onClose()
      onCreated(created)
    },
    onError: (error) => notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' }),
  })
  const trimmed = name.trim()
  return (
    <Modal opened={opened} onClose={onClose} title="Register a worker">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (trimmed) create.mutate(trimmed)
        }}
      >
        <Stack gap="sm">
          <TextInput
            label="Name"
            description="So you can tell the machines apart, e.g. “lab RTX 4090”."
            value={name}
            onChange={(e) => setName(e.currentTarget.value)}
            maxLength={100}
            data-autofocus
          />
          <Group justify="flex-end">
            <Button variant="default" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!trimmed} loading={create.isPending}>
              Register
            </Button>
          </Group>
        </Stack>
      </form>
    </Modal>
  )
}

export function WorkersPage() {
  const queryClient = useQueryClient()
  const { data, isLoading, isError, refetch } = useQuery({
    ...getAdminListWorkersQueryOptions(),
    refetchInterval: 5000,
  })
  const [registering, setRegistering] = useState(false)
  const [created, setCreated] = useState<CreatedWorker | null>(null)

  const refresh = () => queryClient.invalidateQueries({ queryKey: getAdminListWorkersQueryKey() })
  const onError = (error: unknown) =>
    notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' })

  const toggle = useMutation({
    mutationFn: (w: WorkerOut) => adminUpdateWorker(w.id, { enabled: !w.enabled }),
    onSuccess: refresh,
    onError,
  })
  const revoke = useMutation({
    mutationFn: (id: string) => adminRevokeWorker(id),
    onSuccess: () => {
      refresh()
      notifications.show({ title: 'Worker revoked', message: 'Its token no longer works.', color: 'gray' })
    },
    onError,
  })

  const confirmRevoke = (w: WorkerOut) =>
    modals.openConfirmModal({
      title: 'Revoke worker',
      children: (
        <Text size="sm">
          Revoke “{w.name}”? Its token stops working for good.
          {w.currentRunId && ` The run it is training (“${w.currentRunName}”) is failed now.`}
        </Text>
      ),
      labels: { confirm: 'Revoke', cancel: 'Keep' },
      confirmProps: { color: 'red' },
      onConfirm: () => revoke.mutate(w.id),
    })

  const columns: DataTableColumn<WorkerOut>[] = [
    {
      key: 'worker',
      header: 'Worker',
      render: (w) => (
        <div>
          <Text size="sm" fw={600}>
            {w.name}
          </Text>
          <Text size="xs" c="dimmed">
            {[w.hostname, w.gpu].filter(Boolean).join(' · ') || 'Has not reported in yet'}
            {w.version ? ` · omnitrain ${w.version}` : ''}
          </Text>
        </div>
      ),
    },
    {
      key: 'state',
      header: 'State',
      fit: true,
      render: (w) => <StatusBadge value={stateOf(w)} colorMap={STATE_COLORS} />,
    },
    {
      key: 'training',
      header: 'Training',
      render: (w) =>
        w.currentRunName ? (
          <Text size="sm">{w.currentRunName}</Text>
        ) : (
          <Text size="sm" c="dimmed">
            —
          </Text>
        ),
    },
    {
      key: 'backends',
      header: 'Backends',
      render: (w) =>
        w.backends.length ? (
          <Group gap={4}>
            {w.backends.map((b) => (
              <Badge key={b} size="sm" variant="outline" color="gray">
                {b}
              </Badge>
            ))}
          </Group>
        ) : (
          <Text size="sm" c="dimmed">
            —
          </Text>
        ),
    },
    {
      key: 'seen',
      header: 'Last seen',
      fit: true,
      render: (w) => (
        <Text size="sm" c="dimmed">
          {when(w.lastSeenAt)}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (w) =>
        !w.revokedAt && (
          <Group gap={4} wrap="nowrap">
            <Button
              size="xs"
              variant="subtle"
              leftSection={<PowerIcon size={14} />}
              loading={toggle.isPending && toggle.variables?.id === w.id}
              onClick={() => toggle.mutate(w)}
            >
              {w.enabled ? 'Switch off' : 'Switch on'}
            </Button>
            <Button
              size="xs"
              variant="subtle"
              color="red"
              leftSection={<TrashIcon size={14} />}
              onClick={() => confirmRevoke(w)}
            >
              Revoke
            </Button>
          </Group>
        ),
    },
  ]

  return (
    <div className="flex flex-col gap-2">
      <PageHeader
        title="Workers"
        description="Remote GPU machines that train runs for this server. A new run goes to an idle worker that can train it, otherwise it trains here."
        actions={
          <Button size="xs" leftSection={<PlusIcon size={14} />} onClick={() => setRegistering(true)}>
            Register worker
          </Button>
        }
      />
      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        <DataTable
          columns={columns}
          data={data?.workers ?? []}
          getRowKey={(w) => w.id}
          emptyMessage="No workers yet. Runs train on this server."
        />
      </QueryBoundary>
      <RegisterModal opened={registering} onClose={() => setRegistering(false)} onCreated={setCreated} />
      <TokenModal created={created} onClose={() => setCreated(null)} />
    </div>
  )
}

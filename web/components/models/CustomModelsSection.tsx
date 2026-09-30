/**
 * The custom-model table, shared by "My models" (a user's own) and Admin > Models (everyone's).
 * `api` decides which; the admin variant additionally shows owners, filters, and creates GLOBAL models.
 *
 * The list refreshes on its own while any model is queued or being validated, so a model turns Ready (or
 * Failed, with the reason) without a reload.
 */

import { Badge, Button, Group, Modal, Progress, Select, Stack, Switch, Text, TextInput, Tooltip } from '@mantine/core'
import { useDebouncedValue, useDisclosure } from '@mantine/hooks'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  ArrowClockwiseIcon,
  CloudArrowUpIcon,
  CubeIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  TrashIcon,
} from '@phosphor-icons/react'
import { DataTable, type DataTableColumn, EmptyState, QueryBoundary } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import type { CustomModelOut } from '@public/lib/api/generated/models'
import {
  CUSTOM_MODELS_KEY,
  formatBytes,
  isValidating,
  MODEL_STATUS_COLORS,
  MODEL_STATUS_LABELS,
  type ModelFilters,
  type ModelsApi,
  taskLabel,
} from '@public/lib/customModels'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { CustomModelForm } from './CustomModelForm'
import { ModelFileDropzone } from './ModelFileDropzone'
import { useModelUpload } from './useModelUpload'

const STATUS_OPTIONS = Object.entries(MODEL_STATUS_LABELS).map(([value, label]) => ({ value, label }))

/** A model waiting for its file (its earlier upload was interrupted, or was never started). */
function UploadFileModal({
  api,
  model,
  onClose,
}: {
  api: ModelsApi
  model: CustomModelOut | null
  onClose: () => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const { upload, progress } = useModelUpload(api)
  const busy = progress !== null

  const send = useMutation({
    mutationFn: () => upload(model?.id ?? '', file as File),
    onSuccess: () => {
      notifications.show({ title: 'Uploaded', message: 'The model will be checked now.', color: 'teal' })
      setFile(null)
      onClose()
    },
    onError: (error) => notifications.show({ title: 'Upload failed', message: apiErrorMessage(error), color: 'red' }),
  })

  return (
    <Modal
      opened={!!model}
      onClose={busy ? () => undefined : onClose}
      title={model ? `Upload files for “${model.name}”` : ''}
      centered
    >
      <Stack gap="sm">
        <ModelFileDropzone file={file} onChange={setFile} disabled={busy} />
        {progress !== null && <Progress value={progress * 100} animated />}
        <Group justify="flex-end">
          <Button variant="subtle" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={() => send.mutate()} disabled={!file} loading={send.isPending}>
            Upload
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}

export function CustomModelsSection({ api }: { api: ModelsApi }) {
  const queryClient = useQueryClient()
  const isAdmin = api.audience === 'admin'

  const [scope, setScope] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [q] = useDebouncedValue(search.trim(), 250)
  const [formOpened, { open: openForm, close: closeForm }] = useDisclosure(false)
  const [uploading, setUploading] = useState<CustomModelOut | null>(null)

  const filters: ModelFilters = {
    scope: (scope as ModelFilters['scope']) ?? undefined,
    status: status ?? undefined,
    q: q || undefined,
  }

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: [...CUSTOM_MODELS_KEY, api.audience, filters],
    queryFn: () => api.list(filters),
    refetchInterval: (query) => (query.state.data?.some(isValidating) ? 3000 : false),
    placeholderData: (previous) => previous,
  })
  const models = data ?? []

  const refresh = () => queryClient.invalidateQueries({ queryKey: CUSTOM_MODELS_KEY })
  const onError = (error: unknown) =>
    notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' })

  const toggle = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.update(id, { enabled }),
    onSuccess: refresh,
    onError,
  })
  const retry = useMutation({ mutationFn: (id: string) => api.retry(id), onSuccess: refresh, onError })
  const remove = useMutation({
    mutationFn: (id: string) => api.remove(id),
    onSuccess: (result) => {
      refresh()
      notifications.show({
        title: result.archived ? 'Model archived' : 'Model deleted',
        message: result.archived
          ? 'Runs trained on it still work, so it was hidden rather than deleted.'
          : 'It and its files were removed.',
        color: 'gray',
      })
    },
    onError,
  })

  const confirmRemove = (m: CustomModelOut) =>
    modals.openConfirmModal({
      title: 'Delete model',
      children: (
        <Text size="sm">
          {m.runCount > 0
            ? `${m.runCount} training run${m.runCount === 1 ? '' : 's'} used “${m.name}”, so it will be archived instead of deleted: hidden from new experiments, while those runs keep working.`
            : `Delete “${m.name}” and its files? This cannot be undone.`}
        </Text>
      ),
      labels: { confirm: m.runCount > 0 ? 'Archive' : 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: () => remove.mutate(m.id),
    })

  const columns: DataTableColumn<CustomModelOut>[] = [
    {
      key: 'name',
      header: 'Model',
      render: (m) => (
        <div style={{ minWidth: 0 }}>
          <Text size="sm" fw={600}>
            {m.name}
          </Text>
          <Text size="xs" c="dimmed">
            {m.kind} · {m.sourceKind === 'hub' ? (m.sourceRef ?? 'Hub') : 'uploaded'}
            {m.revision ? ` @ ${m.revision.slice(0, 7)}` : ''}
            {m.description ? ` · ${m.description}` : ''}
          </Text>
        </div>
      ),
    },
    ...(isAdmin
      ? [
          {
            key: 'scope',
            header: 'Owner',
            fit: true,
            render: (m: CustomModelOut) =>
              m.scope === 'global' ? (
                <Badge variant="light" color="grape">
                  Global
                </Badge>
              ) : (
                <Text size="xs" c="dimmed">
                  {m.ownerEmail ?? 'private'}
                </Text>
              ),
          } satisfies DataTableColumn<CustomModelOut>,
        ]
      : []),
    {
      key: 'tasks',
      header: 'Tasks',
      render: (m) => (
        <Group gap={4}>
          {m.tasks.slice(0, 2).map((t) => (
            <Badge key={t} size="xs" variant="outline" color="gray">
              {taskLabel(t)}
            </Badge>
          ))}
          {m.tasks.length > 2 && (
            <Tooltip label={m.tasks.slice(2).map(taskLabel).join(', ')}>
              <Badge size="xs" variant="outline" color="gray">
                +{m.tasks.length - 2}
              </Badge>
            </Tooltip>
          )}
        </Group>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (m) => (
        <div>
          <Badge variant="light" color={MODEL_STATUS_COLORS[m.status] ?? 'gray'}>
            {MODEL_STATUS_LABELS[m.status] ?? m.status}
          </Badge>
          {m.status === 'failed' && m.lastError && (
            <Text size="xs" c="red" lineClamp={2} title={m.lastError} maw={280}>
              {m.lastError}
            </Text>
          )}
        </div>
      ),
    },
    { key: 'size', header: 'Size', fit: true, render: (m) => <Text size="sm">{formatBytes(m.sizeBytes)}</Text> },
    { key: 'runs', header: 'Runs', fit: true, render: (m) => <Text size="sm">{m.runCount}</Text> },
    {
      key: 'enabled',
      header: 'Enabled',
      fit: true,
      render: (m) => (
        <Switch
          checked={m.enabled}
          aria-label={`${m.name} enabled`}
          disabled={toggle.isPending}
          onChange={(e) => toggle.mutate({ id: m.id, enabled: e.currentTarget.checked })}
        />
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (m) => (
        <Group gap={4} wrap="nowrap">
          {m.status === 'pending_upload' && (
            <Button
              size="xs"
              variant="subtle"
              leftSection={<CloudArrowUpIcon size={14} />}
              onClick={() => setUploading(m)}
            >
              Upload file
            </Button>
          )}
          {m.status === 'failed' && (
            <Button
              size="xs"
              variant="subtle"
              leftSection={<ArrowClockwiseIcon size={14} />}
              loading={retry.isPending && retry.variables === m.id}
              onClick={() => retry.mutate(m.id)}
            >
              Retry
            </Button>
          )}
          <Button
            size="xs"
            variant="subtle"
            color="red"
            leftSection={<TrashIcon size={14} />}
            disabled={m.status === 'validating'}
            onClick={() => confirmRemove(m)}
          >
            Delete
          </Button>
        </Group>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-2">
      <Group justify="space-between">
        <Group gap="xs">
          {isAdmin && (
            <>
              <TextInput
                size="xs"
                placeholder="Search name"
                leftSection={<MagnifyingGlassIcon size={14} />}
                value={search}
                onChange={(e) => setSearch(e.currentTarget.value)}
                w={200}
              />
              <Select
                size="xs"
                placeholder="All owners"
                clearable
                value={scope}
                onChange={setScope}
                data={[
                  { value: 'global', label: 'Global' },
                  { value: 'private', label: 'Private' },
                ]}
                w={130}
              />
              <Select
                size="xs"
                placeholder="Any status"
                clearable
                value={status}
                onChange={setStatus}
                data={STATUS_OPTIONS}
                w={150}
              />
            </>
          )}
        </Group>
        <Button leftSection={<PlusIcon size={14} />} onClick={openForm}>
          {isAdmin ? 'New global model' : 'New model'}
        </Button>
      </Group>

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {models.length === 0 ? (
          <EmptyState
            icon={CubeIcon}
            title={isAdmin ? 'No custom models' : 'No models of your own yet'}
            description={
              isAdmin
                ? 'Add a global model to make it available to every user, or wait for users to add their own.'
                : 'Bring a model from the Hugging Face Hub or upload your own weights, then pick it when you train.'
            }
          />
        ) : (
          <DataTable columns={columns} data={models} getRowKey={(m) => m.id} />
        )}
      </QueryBoundary>

      <CustomModelForm
        api={api}
        opened={formOpened}
        onClose={closeForm}
        title={isAdmin ? 'New global model' : 'New model'}
      />
      <UploadFileModal api={api} model={uploading} onClose={() => setUploading(null)} />
    </div>
  )
}

/**
 * API Keys page – issue/list/revoke the bearer credentials the hosted
 * prediction API accepts (POST /api/v1/predict/:runId — see
 * server/routes/api-v1.ts). Only a hash of a key is ever stored server-side
 * (server/db/schema.ts's apiKeys.keyHash), so the raw value is shown
 * exactly once, right after creation — losing it means generating a new one.
 */

import { Alert, Badge, Button, Code, CopyButton, Group, Modal, Stack, Text, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { CheckIcon, CopyIcon, KeyIcon, PlusIcon, TrashIcon, WarningCircleIcon } from '@phosphor-icons/react'
import {
  confirmDelete,
  DataTable,
  type DataTableColumn,
  EmptyState,
  QueryBoundary,
  SectionLabel,
} from '@public/components/ui'
import {
  getCreateApiKeyMutationOptions,
  getListApiKeysQueryKey,
  getListApiKeysQueryOptions,
  revokeApiKey,
} from '@public/lib/api/generated/api-keys/api-keys'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

interface ApiKeySummary {
  id: string
  name: string
  keyPrefix: string
  lastUsedAt: string | Date | null
  createdAt: string | Date
  revokedAt: string | Date | null
}

/** Issue, list and revoke API keys. Rendered as a section of the Settings page. */
export function ApiKeysSection() {
  const queryClient = useQueryClient()
  const keysQueryKey = getListApiKeysQueryKey()

  const { data, isLoading, isError, refetch } = useQuery(getListApiKeysQueryOptions())
  const keys: ApiKeySummary[] = (data as { keys: ApiKeySummary[] } | undefined)?.keys ?? []

  const [createOpened, { open: openCreate, close: closeCreateModal }] = useDisclosure(false)
  const [newKey, setNewKey] = useState<string | null>(null)

  const closeCreate = () => {
    closeCreateModal()
    setNewKey(null)
    form.reset()
  }

  const form = useForm({
    initialValues: { name: '' },
    validate: { name: (v) => (v.trim().length > 0 ? null : 'Name is required') },
  })

  const createKey = useMutation({
    ...getCreateApiKeyMutationOptions(),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: keysQueryKey })
      setNewKey((result as { key: string }).key)
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to create API key', color: 'red' }),
  })

  const revokeKey = useMutation({
    mutationFn: async (keyId: string) => {
      await revokeApiKey(keyId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keysQueryKey })
      notifications.show({ title: 'Key revoked', message: 'This key can no longer be used.', color: 'gray' })
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to revoke API key', color: 'red' }),
  })

  const handleRevoke = (key: ApiKeySummary) =>
    confirmDelete({
      title: 'Revoke API key',
      message: `Revoke "${key.name}"? Anything using this key will stop working immediately. This cannot be undone.`,
      onConfirm: () => revokeKey.mutate(key.id),
    })

  const columns: DataTableColumn<ApiKeySummary>[] = [
    {
      key: 'name',
      header: 'Name',
      render: (k) => (
        <Text size="sm" fw={600}>
          {k.name}
        </Text>
      ),
    },
    {
      key: 'prefix',
      header: 'Key',
      fit: true,
      render: (k) => <Code>{k.keyPrefix}…</Code>,
    },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (k) => (
        <Badge variant="light" color={k.revokedAt ? 'gray' : 'teal'}>
          {k.revokedAt ? 'Revoked' : 'Active'}
        </Badge>
      ),
    },
    {
      key: 'lastUsedAt',
      header: 'Last Used',
      fit: true,
      render: (k) => (
        <Text size="sm" c="dimmed">
          {k.lastUsedAt ? new Date(k.lastUsedAt).toLocaleString() : 'Never'}
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (k) => (
        <Text size="sm" c="dimmed">
          {new Date(k.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (k) =>
        !k.revokedAt && (
          <Button
            size="xs"
            variant="subtle"
            color="red"
            leftSection={<TrashIcon size={14} />}
            onClick={() => handleRevoke(k)}
          >
            Revoke
          </Button>
        ),
    },
  ]

  return (
    <div className="flex flex-col gap-2">
      <Group justify="space-between">
        <div>
          <SectionLabel>API keys</SectionLabel>
          <Text size="xs" c="dimmed">
            Bearer credentials for the hosted prediction API (POST /api/v1/predict/:runId).
          </Text>
        </div>
        <Button leftSection={<PlusIcon size={14} />} onClick={openCreate}>
          New key
        </Button>
      </Group>

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {keys.length === 0 ? (
          <EmptyState
            icon={KeyIcon}
            title="No API keys yet"
            description="Create one to call the prediction API from a script."
          />
        ) : (
          <DataTable columns={columns} data={keys} getRowKey={(k) => k.id} />
        )}
      </QueryBoundary>

      <Modal opened={createOpened} onClose={closeCreate} title="New API Key" centered>
        {newKey ? (
          <Stack gap="md">
            <Alert icon={<WarningCircleIcon size={16} />} color="yellow" title="Copy this now">
              This key won't be shown again — store it somewhere safe.
            </Alert>
            <Group gap="xs" wrap="nowrap" align="flex-start">
              <Code block style={{ flex: 1, overflowWrap: 'break-word' }}>
                {newKey}
              </Code>
              <CopyButton value={newKey}>
                {({ copied, copy }) => (
                  <Button
                    size="xs"
                    variant="light"
                    color={copied ? 'teal' : 'primary'}
                    onClick={copy}
                    leftSection={copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
                  >
                    {copied ? 'Copied' : 'Copy'}
                  </Button>
                )}
              </CopyButton>
            </Group>
            <Group justify="flex-end">
              <Button onClick={closeCreate}>Done</Button>
            </Group>
          </Stack>
        ) : (
          <form onSubmit={form.onSubmit((values) => createKey.mutate({ data: { name: values.name } }))}>
            <Stack gap="md">
              <TextInput
                label="Name"
                placeholder="e.g. Production script"
                data-autofocus
                {...form.getInputProps('name')}
              />
              <Group justify="flex-end">
                <Button variant="subtle" onClick={closeCreate}>
                  Cancel
                </Button>
                <Button type="submit" loading={createKey.isPending}>
                  Create
                </Button>
              </Group>
            </Stack>
          </form>
        )}
      </Modal>
    </div>
  )
}

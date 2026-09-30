/**
 * The on/off table for one kind of plugin (trainer backends, built-in models, export formats, ...): a switch
 * for "all tasks" and a dialog for per-task overrides. Used by Admin > Plugins (one tab per kind) and by the
 * built-in models tab of Admin > Models.
 *
 * A switch only stops NEW use. Existing runs, exports and projects keep working (see
 * ai_service/theseus/services/plugin_settings.py). Precedence: a task's own setting beats the "all tasks"
 * one, so something can be off everywhere and on for a single task.
 */

import { Badge, Button, Group, Modal, SegmentedControl, Stack, Switch, Text, TextInput } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { MagnifyingGlassIcon, SlidersHorizontalIcon } from '@phosphor-icons/react'
import { DataTable, type DataTableColumn, QueryBoundary } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  adminSetPlugin,
  getAdminListPluginsQueryKey,
  getAdminListPluginsQueryOptions,
} from '@public/lib/api/generated/admin/admin'
import type { PluginEntry } from '@public/lib/api/generated/models'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

type TaskChoice = 'inherit' | 'on' | 'off'

const taskChoice = (t: PluginEntry['tasks'][number]): TaskChoice =>
  !t.overridden ? 'inherit' : t.enabled ? 'on' : 'off'

const GROUP_HEADERS: Record<string, string> = { builtin_model: 'Backend', task: 'Status' }

export function PluginTable({ kind, hint }: { kind: string; hint?: string }) {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<PluginEntry | null>(null)
  const [opened, { open, close }] = useDisclosure(false)

  const { data, isLoading, isError, refetch } = useQuery(getAdminListPluginsQueryOptions())

  const set = useMutation({
    mutationFn: ({ plugin, enabled, task }: { plugin: PluginEntry; enabled: boolean | null; task?: string }) =>
      adminSetPlugin(plugin.kind, plugin.id, { enabled, task: task ?? null }),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: getAdminListPluginsQueryKey() })
      // Keep the open dialog in step with what was just saved.
      setEditing((current) => (current && current.id === result.plugin.id ? result.plugin : current))
    },
    onError: (error) => notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' }),
  })

  const rows = (data?.plugins ?? [])
    .filter((p) => p.kind === kind)
    .filter((p) => !search.trim() || `${p.label} ${p.id}`.toLowerCase().includes(search.trim().toLowerCase()))

  const columns: DataTableColumn<PluginEntry>[] = [
    {
      key: 'name',
      header: 'Name',
      render: (p) => (
        <div>
          <Group gap={6}>
            <Text size="sm" fw={600}>
              {p.label}
            </Text>
            {!p.available && (
              <Badge size="xs" color="orange" variant="light" title={p.unavailableReason ?? undefined}>
                unavailable
              </Badge>
            )}
          </Group>
          <Text size="xs" c="dimmed">
            {p.id}
            {p.description ? ` · ${p.description}` : ''}
          </Text>
        </div>
      ),
    },
    {
      key: 'group',
      header: GROUP_HEADERS[kind] ?? 'Group',
      fit: true,
      render: (p) =>
        p.group ? (
          <Badge variant="light" color="gray">
            {p.group}
          </Badge>
        ) : null,
    },
    {
      key: 'overrides',
      header: 'Per-task overrides',
      fit: true,
      render: (p) => {
        const n = p.tasks.filter((t) => t.overridden).length
        return n > 0 ? (
          <Badge variant="light" color="grape">
            {n}
          </Badge>
        ) : (
          <Text size="sm" c="dimmed">
            none
          </Text>
        )
      },
    },
    {
      key: 'enabled',
      header: 'All tasks',
      fit: true,
      render: (p) => (
        <Switch
          checked={p.enabled}
          aria-label={`${p.label} for all tasks`}
          disabled={set.isPending}
          // Switching back on clears the override instead of storing "enabled = true".
          onChange={(e) => set.mutate({ plugin: p, enabled: e.currentTarget.checked ? null : false })}
        />
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (p) =>
        p.tasks.length > 0 && (
          <Button
            size="xs"
            variant="subtle"
            leftSection={<SlidersHorizontalIcon size={14} />}
            onClick={() => {
              setEditing(p)
              open()
            }}
          >
            Per task
          </Button>
        ),
    },
  ]

  return (
    <div className="flex flex-col gap-2">
      <Group justify="space-between">
        <Text size="xs" c="dimmed">
          {hint}
        </Text>
        <TextInput
          placeholder="Filter"
          size="xs"
          leftSection={<MagnifyingGlassIcon size={14} />}
          value={search}
          onChange={(e) => setSearch(e.currentTarget.value)}
          w={220}
        />
      </Group>

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        <DataTable columns={columns} data={rows} getRowKey={(p) => `${p.kind}:${p.id}`} emptyMessage="Nothing here" />
      </QueryBoundary>

      <Modal opened={opened} onClose={close} title={editing ? `${editing.label}: per task` : ''} centered size="lg">
        {editing && (
          <Stack gap="xs">
            <Text size="xs" c="dimmed">
              “Inherit” follows the all-tasks switch ({editing.enabled ? 'on' : 'off'}). On and Off override it for that
              task alone.
            </Text>
            {editing.tasks.map((t) => (
              <Group key={t.task} justify="space-between" wrap="nowrap">
                <div>
                  <Text size="sm">{t.taskLabel}</Text>
                  <Text size="xs" c="dimmed">
                    Currently {t.enabled ? 'enabled' : 'disabled'}
                  </Text>
                </div>
                <SegmentedControl
                  size="xs"
                  value={taskChoice(t)}
                  disabled={set.isPending}
                  data={[
                    { value: 'inherit', label: 'Inherit' },
                    { value: 'on', label: 'On' },
                    { value: 'off', label: 'Off' },
                  ]}
                  onChange={(v) =>
                    set.mutate({
                      plugin: editing,
                      task: t.task,
                      enabled: v === 'inherit' ? null : v === 'on',
                    })
                  }
                />
              </Group>
            ))}
          </Stack>
        )}
      </Modal>
    </div>
  )
}

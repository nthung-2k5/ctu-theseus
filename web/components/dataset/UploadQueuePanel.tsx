/**
 * The Upload page's right-hand pane for text and tabular tasks: everything staged but not yet sent.
 *
 * The split and class of every staged item stay editable — per row, or across the whole batch — until the
 * user presses Upload, so a wrong split or class never has to be repaired afterwards on the Dataset page.
 * Rows go out through POST /projects/:id/items, which is per-item. (File tasks don't come through here; see
 * FileTreeView and FileUploadBar.)
 *
 * The table is virtualized: a CSV import can stage thousands of rows, and every row carries its own selects.
 */

import { ActionIcon, Badge, Button, Group, Select, Stack, Text, ThemeIcon, Title, Tooltip } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ListChecksIcon, TrashIcon, UploadSimpleIcon, XIcon } from '@phosphor-icons/react'
import { type VirtualColumn, VirtualDataTable } from '@public/components/ui'
import { createItems } from '@public/lib/api/generated/datasets/datasets'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import { invalidateProjectScope } from '@public/lib/queries'
import type { StagedEdit, StagedItem } from '@public/lib/uploadQueue'
import type { LabelClass, SplitType } from '@public/store/types'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'

/** The `items.post` payload for one staged item. */
function toItemPayload(item: StagedItem) {
  const annotations =
    item.targetValue != null
      ? [{ annotationType: 'classification' as const, labelStructured: { value: item.targetValue } }]
      : item.classId
        ? [{ annotationType: 'classification' as const, classId: item.classId }]
        : undefined

  return {
    split: item.split,
    ...(item.text != null ? { textFeatures: { rawText: item.text } } : {}),
    ...(item.featuresJson ? { tabularFeatures: { featuresJson: item.featuresJson } } : {}),
    annotations,
  }
}

export function UploadQueuePanel({
  projectId,
  items,
  classes,
  onEdit,
  onEditAll,
  onRemove,
  onClear,
}: {
  projectId: string
  items: StagedItem[]
  classes: LabelClass[]
  onEdit: (id: string, patch: StagedEdit) => void
  onEditAll: (patch: StagedEdit) => void
  onRemove: (id: string) => void
  onClear: () => void
}) {
  const queryClient = useQueryClient()

  const classOptions = useMemo(() => classes.map((c) => ({ value: c.classId, label: c.name })), [classes])

  const splitCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const item of items) counts[item.split] = (counts[item.split] ?? 0) + 1
    return counts
  }, [items])

  /** Rows the task wants labeled but that still carry no class — worth flagging before upload, not blocking. */
  const unassignedCount = useMemo(
    () => (classes.length === 0 ? 0 : items.filter((i) => !i.classId && i.targetValue == null).length),
    [items, classes.length],
  )

  const upload = useMutation({
    mutationFn: async () => {
      try {
        const data = await createItems(projectId, { items: items.map(toItemPayload) })
        return { uploaded: data.created.length, failed: data.failed.length }
      } catch {
        return { uploaded: 0, failed: items.length }
      }
    },
    onSuccess: ({ uploaded, failed }) => {
      invalidateProjectScope(queryClient, projectId)

      notifications.show(
        failed > 0
          ? {
              title: 'Upload partially complete',
              message: `${uploaded} item(s) added, ${failed} failed`,
              color: 'yellow',
            }
          : { title: 'Upload complete', message: `${uploaded} item(s) added to the pool`, color: 'green' },
      )
      onClear()
    },
    onError: () => {
      notifications.show({ title: 'Upload failed', message: 'Nothing was added to the pool', color: 'red' })
    },
  })

  const columns: VirtualColumn<StagedItem>[] = [
    {
      key: 'item',
      header: 'Item',
      render: (item) => (
        <Stack gap={0}>
          <Text size="xs" fw={500} truncate>
            {item.name}
          </Text>
          {item.detail && (
            <Text size="xs" c="dimmed" truncate>
              {item.detail}
            </Text>
          )}
        </Stack>
      ),
    },
    {
      key: 'split',
      header: 'Split',
      width: 130,
      render: (item) => (
        <Select
          size="xs"
          w="100%"
          data={SPLIT_OPTIONS}
          value={item.split}
          onChange={(value) => onEdit(item.id, { split: (value ?? 'train') as SplitType })}
          allowDeselect={false}
          disabled={upload.isPending}
        />
      ),
    },
    ...(classes.length > 0
      ? [
          {
            key: 'class',
            header: 'Class',
            width: 150,
            render: (item: StagedItem) => (
              <Select
                size="xs"
                w="100%"
                placeholder="Unassigned"
                data={classOptions}
                value={item.classId}
                onChange={(value) => onEdit(item.id, { classId: value })}
                searchable
                clearable
                disabled={upload.isPending}
              />
            ),
          },
        ]
      : []),
    ...(items.some((item) => item.targetValue != null)
      ? [
          {
            key: 'target',
            header: 'Target',
            width: 80,
            render: (item: StagedItem) => <Text size="xs">{item.targetValue ?? '—'}</Text>,
          },
        ]
      : []),
    {
      key: 'remove',
      header: '',
      width: 32,
      render: (item) => (
        <ActionIcon
          size="sm"
          variant="subtle"
          color="red"
          aria-label={`Remove ${item.name}`}
          onClick={() => onRemove(item.id)}
          disabled={upload.isPending}
        >
          <XIcon size={14} />
        </ActionIcon>
      ),
    },
  ]

  return (
    <Stack gap="md" style={{ flex: 1, overflow: 'hidden' }}>
      <Group justify="space-between" wrap="nowrap">
        <Group gap="sm">
          <ThemeIcon size="md" variant="light" color="primary">
            <ListChecksIcon size={18} />
          </ThemeIcon>
          <Title order={5}>Ready to upload</Title>
          {items.length > 0 && (
            <Badge size="sm" variant="light">
              {items.length}
            </Badge>
          )}
        </Group>
        <Group gap="xs" wrap="nowrap">
          {items.length > 0 && (
            <Tooltip label="Discard everything staged — nothing has been uploaded yet">
              <Button
                size="xs"
                variant="subtle"
                color="red"
                leftSection={<TrashIcon size={14} />}
                onClick={onClear}
                disabled={upload.isPending}
              >
                Discard
              </Button>
            </Tooltip>
          )}
          <Button
            leftSection={<UploadSimpleIcon size={14} />}
            onClick={() => upload.mutate()}
            loading={upload.isPending}
            disabled={items.length === 0}
          >
            {items.length > 0 ? `Upload ${items.length} item(s)` : 'Upload'}
          </Button>
        </Group>
      </Group>

      {items.length > 0 && (
        <Group justify="space-between" wrap="nowrap" align="center">
          <Group gap="md">
            {SPLIT_OPTIONS.map(({ value, label }) => (
              <Text key={value} size="xs" c="dimmed">
                {label}:{' '}
                <Text component="span" fw={600} c="var(--mantine-color-text)">
                  {splitCounts[value] ?? 0}
                </Text>
              </Text>
            ))}
            {unassignedCount > 0 && (
              <Text size="xs" c="yellow.7">
                {unassignedCount} without a class
              </Text>
            )}
          </Group>
          <Group gap="xs" wrap="nowrap">
            <Text size="xs" c="dimmed">
              Apply to all
            </Text>
            <Select
              size="xs"
              w={130}
              placeholder="Split"
              data={SPLIT_OPTIONS}
              value={null}
              onChange={(value) => value && onEditAll({ split: value as SplitType })}
              disabled={upload.isPending}
            />
            {classes.length > 0 && (
              <Select
                size="xs"
                w={150}
                placeholder="Class"
                data={classOptions}
                value={null}
                onChange={(value) => value && onEditAll({ classId: value })}
                searchable
                disabled={upload.isPending}
              />
            )}
          </Group>
        </Group>
      )}

      <div style={{ flex: 1, minHeight: 0 }}>
        <VirtualDataTable
          columns={columns}
          data={items}
          getRowKey={(item) => item.id}
          emptyMessage="Nothing staged yet — add items on the left, then review each one's split and class here before uploading."
        />
      </div>
    </Stack>
  )
}

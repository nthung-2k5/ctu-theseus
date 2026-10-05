import { Checkbox, Group, Text } from '@mantine/core'
import { DataTable, type DataTableColumn, StatusBadge } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { ReactNode } from 'react'
import { type GroundTruthSpec, groundTruthKind, groundTruthLabel } from './groundTruth'
import { AugmentedBadge, PreprocessedBadge } from './ItemBadges'
import type { CaptionEditing, DatasetListItem, ItemSelection } from './types'
import { tabularRow } from './types'

/** Text/tabular item table: click a row to open it in the item drawer. */
export const ItemsTable = <T extends DatasetListItem>({
  items,
  modality,
  isLoading,
  spec,
  classNameById,
  emptyMessage,
  onOpen,
  renderActions,
  selection,
  captionEditing,
}: {
  items: T[]
  modality: 'text' | 'tabular'
  isLoading?: boolean
  spec: GroundTruthSpec
  classNameById: Map<string, string>
  emptyMessage: string
  onOpen: (item: T) => void
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  captionEditing?: CaptionEditing
}) => {
  const columns: DataTableColumn<T>[] = []
  const kind = groundTruthKind(spec)

  if (selection) {
    const allSelected = items.length > 0 && items.every((i) => selection.selectedIds.has(i.id))
    const someSelected = items.some((i) => selection.selectedIds.has(i.id))
    columns.push({
      key: 'select',
      header: (
        <Checkbox
          aria-label="Select all items on this page"
          checked={allSelected}
          indeterminate={someSelected && !allSelected}
          onChange={() => {
            selection.onToggleMany(
              items.map((i) => i.id),
              !allSelected,
            )
          }}
          size="xs"
        />
      ),
      fit: true,
      render: (item) => (
        <Checkbox
          aria-label={`Select ${item.externalId ?? 'item'}`}
          checked={selection.selectedIds.has(item.id)}
          // Shift-click arrives as the click that produced this change event.
          onChange={(e) => selection.onToggle(item.id, (e.nativeEvent as MouseEvent).shiftKey === true)}
          // Selecting a row must not also open it.
          onClick={(e) => e.stopPropagation()}
          size="xs"
        />
      ),
    })
  }

  columns.push({
    key: 'externalId',
    header: 'External ID',
    render: (item) => (
      <Group gap={6} wrap="nowrap">
        <Text size="xs">{item.externalId ?? '—'}</Text>
        <PreprocessedBadge item={item} />
        <AugmentedBadge item={item} />
      </Group>
    ),
  })

  if (modality === 'text') {
    columns.push({
      key: 'text',
      header: 'Text',
      render: (item) => (
        <Text size="xs" lineClamp={2} maw={420}>
          {item.textFeatures?.rawText ?? '—'}
        </Text>
      ),
    })
  } else {
    const featureKeys = [...new Set(items.flatMap((i) => Object.keys(tabularRow(i.tabularFeatures?.featuresJson))))]
    for (const key of featureKeys) {
      columns.push({
        key: `feat_${key}`,
        header: key,
        render: (item) => <Text size="xs">{String(tabularRow(item.tabularFeatures?.featuresJson)[key] ?? '—')}</Text>,
      })
    }
  }

  columns.push(
    {
      key: 'split',
      header: 'Split',
      fit: true,
      render: (item) => <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="xs" />,
    },
    {
      key: 'label',
      header: kind === 'number' ? 'Target' : (captionEditing?.label ?? 'Label'),
      fit: kind !== 'text',
      render: (item) => (
        <Text size="xs" c="dimmed" lineClamp={kind === 'text' ? 2 : undefined} maw={kind === 'text' ? 320 : undefined}>
          {groundTruthLabel(item.annotations, spec, classNameById) ?? '—'}
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      render: (item) => (
        <Text size="xs" c="dimmed">
          {new Date(item.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
  )

  if (renderActions) {
    columns.push({
      key: 'actions',
      header: '',
      fit: true,
      // The menu's clicks must not also open the row.
      render: (item) => (
        // biome-ignore lint/a11y/noStaticElementInteractions: only stops the click reaching the row's open handler
        // biome-ignore lint/a11y/useKeyWithClickEvents: same
        <span onClick={(e) => e.stopPropagation()}>{renderActions(item)}</span>
      ),
    })
  }

  return (
    <DataTable
      columns={columns}
      data={items}
      getRowKey={(item) => item.id}
      loading={isLoading}
      emptyMessage={emptyMessage}
      onRowClick={onOpen}
    />
  )
}

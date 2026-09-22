import {
  Badge,
  Box,
  Button,
  Card,
  Center,
  Checkbox,
  Group,
  Image,
  Modal,
  SimpleGrid,
  Text,
  Textarea,
  ThemeIcon,
  Tooltip,
} from '@mantine/core'
import { CheckIcon, SpeakerHighIcon } from '@phosphor-icons/react'
import { DataTable, type DataTableColumn, EmptyState, QueryBoundary, StatusBadge } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { Annotation, Modality } from '@public/store/types'
import { type ReactNode, useState } from 'react'

/**
 * The item shape both the Data page's pool table and the Dataset page's
 * per-version viewer render — a superset of every modality's fields, since
 * `GET /projects/:projectId/items` always joins all four feature tables
 * regardless of the dataset's modality (see server/routes/datasets.ts).
 */
export interface DatasetListItem {
  id: string
  externalId: string | null
  splitType: string
  createdAt: string | Date
  annotations?: Annotation[]
  downloadUrl?: string | null
  textFeatures?: { rawText: string } | null
  visionFeatures?: { width: number; height: number; imageFormat: string | null } | null
  audioFeatures?: { durationSeconds: string | number; sampleRateHz: number } | null
  tabularFeatures?: { featuresJson: unknown } | null
  /** Set on an augmented copy (only snapshots built with augmentation have these). */
  sourceItemId?: string | null
  sourceExternalId?: string | null
  augmentation?: unknown
}

/** "image_horizontal_flip" -> "Horizontal flip": the op ids are `<modality>_<name>`. */
const opLabel = (id: string) => {
  const words = id.split('_').slice(1).join(' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** The ops that produced an augmented copy, from its stored `{copy, ops: [{id, params}]}` record. */
const appliedOps = (item: DatasetListItem): string[] => {
  const ops = (item.augmentation as { ops?: { id: string }[] } | null | undefined)?.ops
  return Array.isArray(ops) ? ops.map((o) => opLabel(o.id)) : []
}

const AugmentedBadge = ({ item }: { item: DatasetListItem }) => {
  if (!item.sourceItemId) return null
  const ops = appliedOps(item)
  return (
    <Tooltip
      label={`Augmented from ${item.sourceExternalId ?? 'an original item'}${ops.length ? ` · ${ops.join(', ')}` : ''}`}
      withArrow
      multiline
      maw={280}
    >
      <Badge size="xs" variant="light" color="grape">
        Augmented
      </Badge>
    </Tooltip>
  )
}

/** Per-item checkbox selection, for pages with batch operations (e.g. the Dataset draft page). */
export interface ItemSelection {
  selectedIds: Set<string>
  onToggle: (itemId: string) => void
  /**
   * Set every id to `selected` in one update. The header checkbox needs this:
   * toggling each id individually *inverts* a mixed selection instead of
   * selecting all of it, which is the opposite of what an indeterminate
   * header checkbox means.
   */
  onToggleMany: (itemIds: string[], selected: boolean) => void
}

const findClassAnnotation = (annotations: Annotation[] | undefined) => annotations?.find((a) => a.classId)
const findTextSequenceAnnotation = (annotations: Annotation[] | undefined) =>
  annotations?.find((a) => a.annotationType === 'text_sequence')

const tabularRow = (featuresJson: unknown): Record<string, unknown> =>
  featuresJson && typeof featuresJson === 'object' ? (featuresJson as Record<string, unknown>) : {}

/**
 * Free-text ground truth (image/audio captioning, ASR) has no shared value to
 * bulk-assign across items the way a class does — every item needs its own
 * caption/transcript — so this is the one annotation type edited per-item,
 * from the detail modal, rather than through the bulk toolbar.
 */
export interface CaptionEditing {
  /** e.g. "Caption" or "Transcript" — used as the field label and placeholder. */
  label: string
  onSave: (itemId: string, existingAnnotationId: string | null, text: string) => Promise<void>
}

const CaptionEditor = ({ item, captionEditing }: { item: DatasetListItem; captionEditing: CaptionEditing }) => {
  const existing = findTextSequenceAnnotation(item.annotations)
  const [text, setText] = useState(existing?.labelTextSequence ?? '')
  const [saving, setSaving] = useState(false)
  // Tracks what's actually persisted, separately from `item.annotations`
  // (a snapshot taken when the modal opened) — so "Save" correctly disables
  // itself right after a successful save instead of staying enabled until
  // the modal is closed and reopened against a refetched item.
  const [savedText, setSavedText] = useState(existing?.labelTextSequence ?? '')

  const handleSave = async () => {
    setSaving(true)
    try {
      await captionEditing.onSave(item.id, existing?.id ?? null, text)
      setSavedText(text)
    } finally {
      setSaving(false)
    }
  }

  const dirty = text !== savedText

  return (
    <Box mt="md">
      <Textarea
        label={captionEditing.label}
        placeholder={`Enter a ${captionEditing.label.toLowerCase()}…`}
        value={text}
        onChange={(e) => setText(e.currentTarget.value)}
        autosize
        minRows={2}
        maxRows={6}
      />
      <Group justify="flex-end" mt="xs">
        <Button size="xs" leftSection={<CheckIcon size={14} />} onClick={handleSave} loading={saving} disabled={!dirty}>
          Save
        </Button>
      </Group>
    </Box>
  )
}

/* ── Media (vision/audio) item detail modal ── */
const MediaItemDetails = ({
  item,
  modality,
  classNameById,
  captionEditing,
}: {
  item: DatasetListItem
  modality: 'vision' | 'audio'
  classNameById: Map<string, string>
  captionEditing?: CaptionEditing
}) => {
  const className = classNameById.get(findClassAnnotation(item.annotations)?.classId ?? '')

  return (
    <Box>
      <Box mb="md">
        {modality === 'vision' ? (
          <Image src={item.downloadUrl ?? undefined} alt="" radius="md" fit="contain" mah={360} fallbackSrc="" />
        ) : (
          item.downloadUrl && (
            // biome-ignore lint/a11y/useMediaCaption: raw audio dataset item has no transcript source
            <audio controls src={item.downloadUrl} style={{ width: '100%' }} />
          )
        )}
      </Box>
      <Box>
        <Group justify="space-between" py={4}>
          <Text size="sm" c="dimmed">
            External ID
          </Text>
          <Text size="sm">{item.externalId ?? '—'}</Text>
        </Group>
        <Group justify="space-between" py={4}>
          <Text size="sm" c="dimmed">
            Split
          </Text>
          <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="sm" />
        </Group>
        {className && (
          <Group justify="space-between" py={4}>
            <Text size="sm" c="dimmed">
              Label
            </Text>
            <Text size="sm">{className}</Text>
          </Group>
        )}
        {item.sourceItemId && (
          <>
            <Group justify="space-between" py={4}>
              <Text size="sm" c="dimmed">
                Augmented from
              </Text>
              <Text size="sm">{item.sourceExternalId ?? 'an original item'}</Text>
            </Group>
            {appliedOps(item).length > 0 && (
              <Group justify="space-between" py={4}>
                <Text size="sm" c="dimmed">
                  Applied
                </Text>
                <Text size="sm">{appliedOps(item).join(', ')}</Text>
              </Group>
            )}
          </>
        )}
        {modality === 'vision' && item.visionFeatures && (
          <Group justify="space-between" py={4}>
            <Text size="sm" c="dimmed">
              Dimensions
            </Text>
            <Text size="sm">
              {item.visionFeatures.width} × {item.visionFeatures.height}
              {item.visionFeatures.imageFormat ? ` · ${item.visionFeatures.imageFormat}` : ''}
            </Text>
          </Group>
        )}
        {modality === 'audio' && item.audioFeatures && (
          <Group justify="space-between" py={4}>
            <Text size="sm" c="dimmed">
              Duration
            </Text>
            <Text size="sm">
              {Number(item.audioFeatures.durationSeconds).toFixed(1)}s · {item.audioFeatures.sampleRateHz}Hz
            </Text>
          </Group>
        )}
        <Group justify="space-between" py={4}>
          <Text size="sm" c="dimmed">
            Created
          </Text>
          <Text size="sm">{new Date(item.createdAt).toLocaleString()}</Text>
        </Group>
      </Box>
      {captionEditing && <CaptionEditor item={item} captionEditing={captionEditing} />}
    </Box>
  )
}

/* ── Media (vision/audio) item grid ── */
const MediaItemsGrid = <T extends DatasetListItem>({
  items,
  modality,
  isLoading,
  classNameById,
  emptyMessage,
  renderActions,
  selection,
  captionEditing,
}: {
  items: T[]
  modality: 'vision' | 'audio'
  isLoading?: boolean
  classNameById: Map<string, string>
  emptyMessage: string
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  captionEditing?: CaptionEditing
}) => {
  const [openItem, setOpenItem] = useState<T | null>(null)

  if (isLoading) return <EmptyState loading compact />
  if (items.length === 0) return <EmptyState description={emptyMessage} compact />

  return (
    <>
      <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5, xl: 6 }} spacing="sm">
        {items.map((item) => {
          const className = classNameById.get(findClassAnnotation(item.annotations)?.classId ?? '')
          const caption = captionEditing ? findTextSequenceAnnotation(item.annotations)?.labelTextSequence : undefined
          const isSelected = selection?.selectedIds.has(item.id) ?? false
          return (
            <Card
              key={item.id}
              withBorder
              p={0}
              radius="md"
              className="card-elevated"
              style={{
                cursor: 'pointer',
                overflow: 'hidden',
                outline: isSelected ? '2px solid var(--mantine-primary-color-5)' : undefined,
                outlineOffset: -2,
              }}
              onClick={() => setOpenItem(item)}
            >
              <Box pos="relative">
                {modality === 'vision' ? (
                  <Image src={item.downloadUrl ?? undefined} alt="" h={110} fit="cover" fallbackSrc="" />
                ) : (
                  <Center h={110} bg="var(--mantine-color-default-hover)">
                    <ThemeIcon size={40} variant="light" color="orange" radius="xl">
                      <SpeakerHighIcon size={22} />
                    </ThemeIcon>
                  </Center>
                )}
                {selection && (
                  <Checkbox
                    checked={isSelected}
                    onChange={() => selection.onToggle(item.id)}
                    onClick={(e) => e.stopPropagation()}
                    pos="absolute"
                    top={6}
                    left={6}
                    size="xs"
                  />
                )}
                {renderActions && (
                  <Box pos="absolute" top={4} right={4} onClick={(e) => e.stopPropagation()}>
                    {renderActions(item)}
                  </Box>
                )}
              </Box>
              <Box p={6}>
                <Text size="xs" truncate>
                  {item.externalId ?? 'Untitled'}
                </Text>
                <Group gap={4} mt={4}>
                  <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="xs" />
                  <AugmentedBadge item={item} />
                  {className && (
                    <Badge size="xs" variant="light" color="teal">
                      {className}
                    </Badge>
                  )}
                </Group>
                {captionEditing && (
                  <Text size="xs" c={caption ? 'dimmed' : 'orange'} lineClamp={2} mt={4}>
                    {caption || `No ${captionEditing.label.toLowerCase()} yet`}
                  </Text>
                )}
              </Box>
            </Card>
          )
        })}
      </SimpleGrid>

      <Modal
        opened={!!openItem}
        onClose={() => setOpenItem(null)}
        title={openItem?.externalId ?? 'Item details'}
        size="lg"
        centered
      >
        {openItem && (
          <MediaItemDetails
            item={openItem}
            modality={modality}
            classNameById={classNameById}
            captionEditing={captionEditing}
          />
        )}
      </Modal>
    </>
  )
}

/* ── Text/tabular item table ── */
const TabularTextItemsTable = <T extends DatasetListItem>({
  items,
  modality,
  isLoading,
  classNameById,
  emptyMessage,
  renderActions,
  selection,
}: {
  items: T[]
  modality: 'text' | 'tabular'
  isLoading?: boolean
  classNameById: Map<string, string>
  emptyMessage: string
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
}) => {
  const columns: DataTableColumn<T>[] = []

  if (selection) {
    const allSelected = items.length > 0 && items.every((i) => selection.selectedIds.has(i.id))
    const someSelected = items.some((i) => selection.selectedIds.has(i.id))
    columns.push({
      key: 'select',
      header: (
        <Checkbox
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
        <Checkbox checked={selection.selectedIds.has(item.id)} onChange={() => selection.onToggle(item.id)} size="xs" />
      ),
    })
  }

  columns.push({
    key: 'externalId',
    header: 'External ID',
    render: (item) => (
      <Group gap={6} wrap="nowrap">
        <Text size="xs">{item.externalId ?? '—'}</Text>
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
      header: 'Label',
      fit: true,
      render: (item) => {
        const classId = findClassAnnotation(item.annotations)?.classId
        return (
          <Text size="xs" c="dimmed">
            {classId ? (classNameById.get(classId) ?? '—') : '—'}
          </Text>
        )
      },
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
      render: renderActions,
    })
  }

  return (
    <DataTable
      columns={columns}
      data={items}
      getRowKey={(item) => item.id}
      loading={isLoading}
      emptyMessage={emptyMessage}
    />
  )
}

/**
 * Dispatches to a thumbnail grid (vision/audio — click a card to open a
 * modal with the full image/audio player and metadata) or a table
 * (text/tabular), by dataset modality. Shared by the Data page's pool
 * browser and the Dataset page's per-version/split item viewer so both
 * present items the same way. Never shows the item's internal id — only
 * externalId, which is what an operator actually recognizes an item by.
 *
 * Pass `selection` to turn on a checkbox per item (grid: card corner, table:
 * leading column with a header "select all this page" checkbox) for pages
 * that need batch operations — the caller owns the selected-id set.
 */
export function ItemsByModality<T extends DatasetListItem>({
  items,
  modality,
  isLoading,
  isError,
  onRetry,
  classNameById,
  emptyMessage = 'No items yet',
  renderActions,
  selection,
  captionEditing,
}: {
  items: T[]
  modality: Modality
  isLoading?: boolean
  isError?: boolean
  onRetry?: () => void
  classNameById: Map<string, string>
  emptyMessage?: string
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  /** Enables per-item free-text caption/transcript editing — vision/audio modalities only (see `CaptionEditor`). */
  captionEditing?: CaptionEditing
}) {
  // A failed fetch must not fall through to `emptyMessage` — "No items in the
  // draft yet" is a very different claim from "the request failed".
  if (isError) return <QueryBoundary isLoading={false} isError onRetry={onRetry} compact />

  if (modality === 'vision' || modality === 'audio') {
    return (
      <MediaItemsGrid
        items={items}
        modality={modality}
        isLoading={isLoading}
        classNameById={classNameById}
        emptyMessage={emptyMessage}
        renderActions={renderActions}
        captionEditing={captionEditing}
        selection={selection}
      />
    )
  }

  return (
    <TabularTextItemsTable
      items={items}
      modality={modality}
      isLoading={isLoading}
      classNameById={classNameById}
      emptyMessage={emptyMessage}
      renderActions={renderActions}
      selection={selection}
    />
  )
}

import { ActionIcon, Drawer, Group, Text } from '@mantine/core'
import { useHotkeys } from '@mantine/hooks'
import { CaretLeftIcon, CaretRightIcon } from '@phosphor-icons/react'
import type { Modality } from '@public/store/types'
import type { GroundTruthSpec } from './groundTruth'
import { ItemDetail } from './ItemDetail'
import type { CaptionEditing, DatasetListItem, ItemEditing } from './types'

/** Mounted only while the drawer is open, so j/k and the arrow keys aren't swallowed the rest of the time. */
const StepHotkeys = ({ onPrev, onNext }: { onPrev?: () => void; onNext?: () => void }) => {
  // useHotkeys skips inputs, textareas and selects by default, so typing a caption or number never steps items.
  useHotkeys([
    ['j', () => onNext?.()],
    ['ArrowRight', () => onNext?.()],
    ['k', () => onPrev?.()],
    ['ArrowLeft', () => onPrev?.()],
  ])
  return null
}

/**
 * The per-item detail view for every modality, with prev/next across the page being browsed.
 * `item` is looked up from the live list by the caller, so edits show up here after the refetch;
 * when the item leaves the list (deleted, or edited out of the current filter) the drawer closes.
 */
export function ItemDrawer({
  item,
  position,
  total,
  onPrev,
  onNext,
  onClose,
  modality,
  spec,
  classNameById,
  captionEditing,
  editing,
}: {
  item: DatasetListItem | null
  /** 1-based position within the current page. */
  position: number
  total: number
  onPrev?: () => void
  onNext?: () => void
  onClose: () => void
  modality: Modality
  spec: GroundTruthSpec
  classNameById: Map<string, string>
  captionEditing?: CaptionEditing
  editing?: ItemEditing
}) {
  return (
    <Drawer
      opened={item !== null}
      onClose={onClose}
      position="right"
      size="md"
      title={
        <Group gap="xs" wrap="nowrap">
          <ActionIcon variant="subtle" color="gray" onClick={onPrev} disabled={!onPrev} aria-label="Previous item (k)">
            <CaretLeftIcon size={16} />
          </ActionIcon>
          <ActionIcon variant="subtle" color="gray" onClick={onNext} disabled={!onNext} aria-label="Next item (j)">
            <CaretRightIcon size={16} />
          </ActionIcon>
          <Text fw={600} truncate maw={220}>
            {item?.externalId ?? 'Item details'}
          </Text>
          <Text size="xs" c="dimmed">
            {position} / {total}
          </Text>
        </Group>
      }
    >
      {item && (
        <>
          <StepHotkeys onPrev={onPrev} onNext={onNext} />
          <ItemDetail
            key={item.id}
            item={item}
            modality={modality}
            spec={spec}
            classNameById={classNameById}
            captionEditing={captionEditing}
            editing={editing}
          />
        </>
      )}
    </Drawer>
  )
}

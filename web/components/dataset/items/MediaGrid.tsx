import {
  Badge,
  Box,
  Card,
  Center,
  Checkbox,
  Group,
  Image,
  SimpleGrid,
  Text,
  ThemeIcon,
  UnstyledButton,
} from '@mantine/core'
import { PlayIcon, SpeakerHighIcon } from '@phosphor-icons/react'
import { EmptyState, StatusBadge, VirtualGrid } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import { type Direction, nextInDirection, type Box as Rect } from '@public/lib/gridNav'
import { Fragment, type KeyboardEvent, type ReactNode, useRef } from 'react'
import { type GroundTruthSpec, groundTruthKind, groundTruthLabel } from './groundTruth'
import { AugmentedBadge, PreprocessedBadge } from './ItemBadges'
import { ItemsSkeleton } from './ItemsSkeleton'
import type { CaptionEditing, DatasetListItem, ItemSelection } from './types'

/**
 * Pages longer than this render through `VirtualGrid`, which mounts only the rows near the viewport. Shorter
 * ones keep the plain grid, so the usual 20-100 per page never goes through the virtualizer at all.
 */
export const VIRTUALIZE_ABOVE = 100

const ARROW_DIRECTIONS: Record<string, Direction> = {
  ArrowLeft: 'left',
  ArrowRight: 'right',
  ArrowUp: 'up',
  ArrowDown: 'down',
}

/** Media (vision/audio/video) item grid: click a card (or press Enter on its name) to open it in the item drawer. */
export const MediaGrid = <T extends DatasetListItem>({
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
  modality: 'vision' | 'audio' | 'video'
  isLoading?: boolean
  spec: GroundTruthSpec
  classNameById: Map<string, string>
  emptyMessage: string
  onOpen: (item: T) => void
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  captionEditing?: CaptionEditing
}) => {
  const containerRef = useRef<HTMLDivElement>(null)

  if (isLoading) return <ItemsSkeleton variant="grid" />
  if (items.length === 0) return <EmptyState description={emptyMessage} compact />

  const kind = groundTruthKind(spec)

  // Arrow keys move between the cards' open buttons by where they are on screen (so it is right at any
  // breakpoint and for the virtualized grid, where only some rows exist); Escape drops the selection.
  const onGridKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement
    if (!target.hasAttribute('data-grid-item')) return
    if (event.key === 'Escape') {
      selection?.onClear?.()
      return
    }
    const direction = ARROW_DIRECTIONS[event.key]
    if (!direction || !containerRef.current) return
    const buttons = [...containerRef.current.querySelectorAll<HTMLElement>('[data-grid-item]')]
    const boxes: Rect[] = buttons.map(
      (b) => b.closest<HTMLElement>('[data-grid-card]')?.getBoundingClientRect() ?? b.getBoundingClientRect(),
    )
    const next = nextInDirection(boxes, buttons.indexOf(target), direction)
    if (next !== null) {
      event.preventDefault()
      buttons[next].focus()
    }
  }

  const renderCard = (item: T) => {
    const label = groundTruthLabel(item.annotations, spec, classNameById)
    const isSelected = selection?.isSelected(item.id) ?? false
    const name = item.externalId ?? 'Untitled'
    return (
      <Card
        data-grid-card
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
        onClick={() => onOpen(item)}
      >
        <Box pos="relative">
          {modality === 'vision' || (modality === 'video' && item.thumbnailUrl) ? (
            <Image
              src={item.thumbnailUrl ?? item.downloadUrl ?? undefined}
              // Decorative: the "Open <name>" button beside it is what names the item.
              alt=""
              h={110}
              fit="cover"
              loading="lazy"
              // A thumbnail that can't be made (or a stale link) falls back to the full-size image. A video has
              // no image to fall back to, so a poster frame that fails just shows the empty placeholder.
              fallbackSrc={modality === 'video' ? '' : (item.downloadUrl ?? '')}
            />
          ) : (
            <Center h={110} bg="var(--mantine-color-default-hover)">
              <ThemeIcon size={40} variant="light" color={modality === 'video' ? 'pink' : 'orange'} radius="xl">
                {modality === 'video' ? <PlayIcon size={22} /> : <SpeakerHighIcon size={22} />}
              </ThemeIcon>
            </Center>
          )}
          {modality === 'video' && item.thumbnailUrl && (
            <Center pos="absolute" inset={0} style={{ pointerEvents: 'none' }}>
              <ThemeIcon size={36} variant="filled" color="dark" radius="xl" style={{ opacity: 0.7 }}>
                <PlayIcon size={18} weight="fill" />
              </ThemeIcon>
            </Center>
          )}
          {selection && (
            <Checkbox
              aria-label={`Select ${name}`}
              checked={isSelected}
              // Shift-click arrives as the click that produced this change event.
              onChange={(e) => selection.onToggle(item.id, (e.nativeEvent as MouseEvent).shiftKey === true)}
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
          {/* The keyboard way in: Enter opens the item, Space ticks it (Shift+Space ticks a range). */}
          <UnstyledButton
            data-grid-item
            className="mantine-focus-auto"
            aria-label={`Open ${name}`}
            aria-pressed={selection ? isSelected : undefined}
            w="100%"
            ta="left"
            onClick={(e) => {
              e.stopPropagation() // the card's own click already opens it
              onOpen(item)
            }}
            onKeyDown={(e) => {
              if (e.key !== ' ' || !selection) return
              e.preventDefault()
              selection.onToggle(item.id, e.shiftKey)
            }}
          >
            <Text size="xs" truncate>
              {name}
            </Text>
          </UnstyledButton>
          <Group gap={4} mt={4}>
            <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="xs" />
            <PreprocessedBadge item={item} />
            <AugmentedBadge item={item} />
            {label && kind !== 'text' && (
              <Badge size="xs" variant="light" color="teal">
                {label}
              </Badge>
            )}
          </Group>
          {kind === 'text' && (
            <Text size="xs" c={label ? 'dimmed' : 'orange'} lineClamp={2} mt={4}>
              {label || `No ${(captionEditing?.label ?? 'label').toLowerCase()} yet`}
            </Text>
          )}
        </Box>
      </Card>
    )
  }

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: not a widget itself; it only routes key presses from the cards' buttons
    <div ref={containerRef} onKeyDown={onGridKeyDown}>
      {items.length > VIRTUALIZE_ABOVE ? (
        <VirtualGrid items={items} getKey={(item) => item.id} renderItem={renderCard} />
      ) : (
        <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5, xl: 6 }} spacing="sm">
          {items.map((item) => (
            <Fragment key={item.id}>{renderCard(item)}</Fragment>
          ))}
        </SimpleGrid>
      )}
    </div>
  )
}

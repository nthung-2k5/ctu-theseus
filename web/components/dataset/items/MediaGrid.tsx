import { Badge, Box, Card, Center, Checkbox, Group, Image, SimpleGrid, Text, ThemeIcon } from '@mantine/core'
import { SpeakerHighIcon } from '@phosphor-icons/react'
import { EmptyState, StatusBadge } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { ReactNode } from 'react'
import { type GroundTruthSpec, groundTruthKind, groundTruthLabel } from './groundTruth'
import { AugmentedBadge, PreprocessedBadge } from './ItemBadges'
import type { CaptionEditing, DatasetListItem, ItemSelection } from './types'

/** Media (vision/audio) item grid: click a card to open it in the item drawer. */
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
  modality: 'vision' | 'audio'
  isLoading?: boolean
  spec: GroundTruthSpec
  classNameById: Map<string, string>
  emptyMessage: string
  onOpen: (item: T) => void
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  captionEditing?: CaptionEditing
}) => {
  if (isLoading) return <EmptyState loading compact />
  if (items.length === 0) return <EmptyState description={emptyMessage} compact />

  const kind = groundTruthKind(spec)

  return (
    <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5, xl: 6 }} spacing="sm">
      {items.map((item) => {
        const label = groundTruthLabel(item.annotations, spec, classNameById)
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
            onClick={() => onOpen(item)}
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
              <Text size="xs" truncate>
                {item.externalId ?? 'Untitled'}
              </Text>
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
      })}
    </SimpleGrid>
  )
}

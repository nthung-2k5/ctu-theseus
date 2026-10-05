import { Badge, Box, Card, Center, Checkbox, Group, Image, Modal, SimpleGrid, Text, ThemeIcon } from '@mantine/core'
import { SpeakerHighIcon } from '@phosphor-icons/react'
import { EmptyState, StatusBadge } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import { type ReactNode, useState } from 'react'
import { AugmentedBadge, PreprocessedBadge } from './ItemBadges'
import { ItemDetail } from './ItemDetail'
import {
  type CaptionEditing,
  type DatasetListItem,
  findClassAnnotation,
  findTextSequenceAnnotation,
  type ItemSelection,
} from './types'

/** Media (vision/audio) item grid: click a card for the full image/audio player and metadata. */
export const MediaGrid = <T extends DatasetListItem>({
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
          <ItemDetail
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

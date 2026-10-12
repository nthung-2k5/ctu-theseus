import { Box, Button, Code, Group, Image, ScrollArea, Select, Table, Text } from '@mantine/core'
import { TrashIcon } from '@phosphor-icons/react'
import { confirmDelete, StatusBadge } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { Modality } from '@public/store/types'
import type { ReactNode } from 'react'
import { SPLIT_TYPES } from '../VersionBrowsing'
import { BoxLabelling } from './BoxLabelling'
import { GroundTruthEditor } from './GroundTruthEditor'
import { type GroundTruthSpec, groundTruthKind } from './groundTruth'
import { appliedOps } from './ops'
import { type CaptionEditing, type DatasetListItem, type ItemEditing, tabularRow } from './types'

const Row = ({ label, children }: { label: string; children: ReactNode }) => (
  <Group justify="space-between" py={4}>
    <Text size="sm" c="dimmed">
      {label}
    </Text>
    {children}
  </Group>
)

/** The item's own content, by modality: the media itself, the full text, or the record's columns. */
const Preview = ({ item, modality }: { item: DatasetListItem; modality: Modality }) => {
  if (modality === 'vision') {
    return (
      <Image
        src={item.downloadUrl ?? undefined}
        alt={item.externalId ?? ''}
        radius="md"
        fit="contain"
        mah={360}
        fallbackSrc=""
      />
    )
  }
  if (modality === 'audio') {
    return item.downloadUrl ? (
      // biome-ignore lint/a11y/useMediaCaption: raw audio dataset item has no transcript source
      <audio controls src={item.downloadUrl} style={{ width: '100%' }} />
    ) : null
  }
  if (modality === 'video') {
    return item.downloadUrl ? (
      // biome-ignore lint/a11y/useMediaCaption: raw video dataset item has no caption track
      <video
        controls
        preload="metadata"
        src={item.downloadUrl}
        style={{ width: '100%', maxHeight: 360, borderRadius: 8, background: '#000' }}
      />
    ) : null
  }
  if (modality === 'text') {
    return (
      <ScrollArea.Autosize mah={260}>
        <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
          {item.textFeatures?.rawText ?? '—'}
        </Text>
      </ScrollArea.Autosize>
    )
  }
  const record = Object.entries(tabularRow(item.tabularFeatures?.featuresJson))
  return (
    <ScrollArea.Autosize mah={260}>
      <Table withTableBorder verticalSpacing={4}>
        <Table.Tbody>
          {record.map(([key, value]) => (
            <Table.Tr key={key}>
              <Table.Th w="40%">{key}</Table.Th>
              <Table.Td>
                {value !== null && typeof value === 'object' ? (
                  <Code>{JSON.stringify(value)}</Code>
                ) : (
                  String(value ?? '—')
                )}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea.Autosize>
  )
}

/**
 * Everything the drawer shows for one item. Key it by item id: the editors below hold draft state
 * that must not leak from one item to the next when the user steps through the page.
 */
export const ItemDetail = ({
  item,
  modality,
  spec,
  classNameById,
  classColorById,
  captionEditing,
  editing,
}: {
  item: DatasetListItem
  modality: Modality
  spec: GroundTruthSpec
  classNameById: Map<string, string>
  classColorById?: Map<string, string>
  captionEditing?: CaptionEditing
  editing?: ItemEditing
}) => {
  const preprocessed = appliedOps(item.preprocessing)
  const augmented = appliedOps(item.augmentation)
  // Boxes are drawn ON the image and edited there, so the picture and its labels are one piece, not two sections.
  const drawsBoxes = modality === 'vision' && groundTruthKind(spec) === 'boxes'

  return (
    <Box>
      {drawsBoxes ? (
        <Box mb="md">
          <BoxLabelling
            item={item}
            classNameById={classNameById}
            classColorById={classColorById}
            editing={editing?.boxes}
          />
        </Box>
      ) : (
        <>
          <Box mb="md">
            <Preview item={item} modality={modality} />
          </Box>

          <Box mb="md">
            <GroundTruthEditor
              item={item}
              spec={spec}
              classNameById={classNameById}
              captionEditing={captionEditing}
              editing={editing}
            />
          </Box>
        </>
      )}

      <Box>
        {modality !== 'text' && (
          <Row label="External ID">
            <Text size="sm">{item.externalId ?? '—'}</Text>
          </Row>
        )}
        <Row label="Split">
          {editing ? (
            <Select
              size="xs"
              w={130}
              aria-label="Split"
              data={SPLIT_TYPES.map((s) => ({ value: s, label: s[0].toUpperCase() + s.slice(1) }))}
              value={item.splitType}
              allowDeselect={false}
              onChange={(next) =>
                next && next !== item.splitType && editing.onSetSplit(item.id, next as (typeof SPLIT_TYPES)[number])
              }
            />
          ) : (
            <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="sm" />
          )}
        </Row>
        {item.sourceItemId && (
          <>
            <Row label="Derived from">
              <Text size="sm">{item.sourceExternalId ?? 'an original item'}</Text>
            </Row>
            {preprocessed.length > 0 && (
              <Row label="Preprocessed">
                <Text size="sm">{preprocessed.join(', ')}</Text>
              </Row>
            )}
            {augmented.length > 0 && (
              <Row label="Augmented">
                <Text size="sm">{augmented.join(', ')}</Text>
              </Row>
            )}
          </>
        )}
        {modality === 'vision' && item.visionFeatures && (
          <Row label="Dimensions">
            <Text size="sm">
              {item.visionFeatures.width} × {item.visionFeatures.height}
              {item.visionFeatures.imageFormat ? ` · ${item.visionFeatures.imageFormat}` : ''}
            </Text>
          </Row>
        )}
        {modality === 'audio' && item.audioFeatures && (
          <Row label="Duration">
            <Text size="sm">
              {Number(item.audioFeatures.durationSeconds).toFixed(1)}s · {item.audioFeatures.sampleRateHz}Hz
            </Text>
          </Row>
        )}
        {modality === 'video' && item.videoFeatures && (
          <>
            <Row label="Dimensions">
              <Text size="sm">
                {item.videoFeatures.width} × {item.videoFeatures.height} · {Number(item.videoFeatures.fps).toFixed(1)}{' '}
                fps
              </Text>
            </Row>
            <Row label="Duration">
              <Text size="sm">
                {Number(item.videoFeatures.durationSeconds).toFixed(1)}s · {item.videoFeatures.frameCount} frames
              </Text>
            </Row>
          </>
        )}
        <Row label="Created">
          <Text size="sm">{new Date(item.createdAt).toLocaleString()}</Text>
        </Row>
      </Box>

      {editing && (
        <Group justify="flex-end" mt="lg">
          <Button
            size="xs"
            variant="light"
            color="red"
            leftSection={<TrashIcon size={12} />}
            onClick={() =>
              confirmDelete({
                title: 'Delete item',
                message: `Remove ${item.externalId ?? 'this item'} from the draft? An item still referenced by a snapshot is kept for that snapshot.`,
                onConfirm: () => editing.onDelete(item.id),
              })
            }
          >
            Delete
          </Button>
        </Group>
      )}
    </Box>
  )
}

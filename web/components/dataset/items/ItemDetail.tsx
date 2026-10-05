import { Box, Group, Image, Text } from '@mantine/core'
import { StatusBadge } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { ReactNode } from 'react'
import { CaptionEditor } from './CaptionEditor'
import { appliedOps } from './ops'
import { type CaptionEditing, type DatasetListItem, findClassAnnotation } from './types'

const Row = ({ label, children }: { label: string; children: ReactNode }) => (
  <Group justify="space-between" py={4}>
    <Text size="sm" c="dimmed">
      {label}
    </Text>
    {children}
  </Group>
)

/** Media (vision/audio) item detail, shown in the modal opened from the grid. */
export const ItemDetail = ({
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
  const preprocessed = appliedOps(item.preprocessing)
  const augmented = appliedOps(item.augmentation)

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
        <Row label="External ID">
          <Text size="sm">{item.externalId ?? '—'}</Text>
        </Row>
        <Row label="Split">
          <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="sm" />
        </Row>
        {className && (
          <Row label="Label">
            <Text size="sm">{className}</Text>
          </Row>
        )}
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
        <Row label="Created">
          <Text size="sm">{new Date(item.createdAt).toLocaleString()}</Text>
        </Row>
      </Box>
      {captionEditing && <CaptionEditor item={item} captionEditing={captionEditing} />}
    </Box>
  )
}

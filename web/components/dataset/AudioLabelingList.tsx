import { Group, Stack, Text } from '@mantine/core'
import type { Annotation, LabelClass } from '@public/store/types'
import { LabelingList } from './LabelingList'

interface AudioItem {
  id: string
  externalId: string | null
  downloadUrl: string | null
  annotations?: Annotation[]
  audioFeatures?: { durationSeconds: string | number; sampleRateHz: number } | null
}

export function AudioLabelingList<T extends AudioItem>({
  projectId,
  items,
  classes,
  isLoading,
}: {
  projectId: string
  items: T[]
  classes: LabelClass[]
  isLoading?: boolean
}) {
  return (
    <LabelingList
      projectId={projectId}
      items={items}
      classes={classes}
      isLoading={isLoading}
      contentHeader="Audio"
      renderContent={(item) => (
        <Stack gap={2}>
          <Text size="xs" c="dimmed" truncate="end" maw={220}>
            {item.externalId ?? item.id.slice(0, 8)}
          </Text>
          {item.downloadUrl && (
            // biome-ignore lint/a11y/useMediaCaption: labeling audio has no transcript source
            <audio controls src={item.downloadUrl} style={{ height: 32, width: 240 }} />
          )}
          {item.audioFeatures && (
            <Group gap={6}>
              <Text size="xs" c="dimmed">
                {Number(item.audioFeatures.durationSeconds).toFixed(1)}s
              </Text>
              <Text size="xs" c="dimmed">
                {item.audioFeatures.sampleRateHz}Hz
              </Text>
            </Group>
          )}
        </Stack>
      )}
    />
  )
}

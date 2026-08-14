import { Text } from '@mantine/core'
import type { Annotation, LabelClass } from '@public/store/types'
import { LabelingList } from './LabelingList'

interface TextItem {
  id: string
  annotations?: Annotation[]
  textFeatures?: { rawText: string } | null
}

export function TextLabelingList<T extends TextItem>({
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
      contentHeader="Text"
      renderContent={(item) => (
        <Text size="sm" lineClamp={2} maw={480}>
          {item.textFeatures?.rawText ?? '—'}
        </Text>
      )}
    />
  )
}

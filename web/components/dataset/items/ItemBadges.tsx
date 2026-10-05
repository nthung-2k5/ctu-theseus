import { Badge, Tooltip } from '@mantine/core'
import { appliedOps } from './ops'
import type { DatasetListItem } from './types'

export const PreprocessedBadge = ({ item }: { item: DatasetListItem }) => {
  if (!item.preprocessing) return null
  const ops = appliedOps(item.preprocessing)
  return (
    <Tooltip label={`Preprocessed${ops.length ? ` · ${ops.join(', ')}` : ''}`} withArrow multiline maw={280}>
      <Badge size="xs" variant="light" color="cyan">
        Preprocessed
      </Badge>
    </Tooltip>
  )
}

export const AugmentedBadge = ({ item }: { item: DatasetListItem }) => {
  if (!item.augmentation) return null
  const ops = appliedOps(item.augmentation)
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

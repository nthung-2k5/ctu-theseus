import { Badge, Tooltip } from '@mantine/core'
import type { Attention } from '@public/lib/classDraft'

/** Why a class needs a look, with the consequence in the tooltip so the badge is never just a label. */
export function AttentionBadge({ attention, min }: { attention: Attention | null; min: number }) {
  if (!attention) return null
  const { text, color, why } =
    attention.kind === 'empty'
      ? {
          text: 'empty',
          color: 'red',
          why: 'No items in the draft have this class, so a model has nothing to learn it from.',
        }
      : attention.kind === 'few'
        ? {
            text: `<${min} items`,
            color: 'yellow',
            why: `Auto-split can't spread a class with fewer than ${min} items across splits, so it is skipped when stratifying.`,
          }
        : {
            text: `no ${attention.splits.join(' / ')}`,
            color: 'yellow',
            why: `Items of this class are missing from ${attention.splits.join(' and ')}, so the model can't be checked on it there.`,
          }
  return (
    <Tooltip label={why} multiline maw={280} withArrow>
      <Badge size="xs" variant="light" color={color} tt="none" style={{ cursor: 'help', flexShrink: 0 }}>
        {text}
      </Badge>
    </Tooltip>
  )
}

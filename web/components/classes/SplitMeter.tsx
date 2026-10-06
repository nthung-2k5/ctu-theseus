import { Tooltip } from '@mantine/core'
import { type ClassCounts, SPLITS } from '@public/lib/classDraft'
import { SPLIT_COLORS } from '@public/lib/constants'

/**
 * One class's item count as a bar: its length is the class's share of the largest class, and it is cut into
 * train / validation / test segments, so imbalance between classes and a class missing from a split are both
 * visible at a glance. An empty class is just an empty track, not a missing bar.
 */
export function SplitMeter({ counts, max, height = 8 }: { counts: ClassCounts; max: number; height?: number }) {
  const width = max > 0 ? Math.min(100, (counts.total / max) * 100) : 0
  const label = SPLITS.map((s) => `${s} ${counts[s]}`).join(', ')
  return (
    <Tooltip label={label} withArrow>
      <div
        role="img"
        aria-label={`${counts.total} items: ${label}`}
        style={{
          width: '100%',
          height,
          borderRadius: height / 2,
          background: 'var(--mantine-color-default-border)',
          overflow: 'hidden',
        }}
      >
        <div style={{ display: 'flex', width: `${width}%`, height: '100%' }}>
          {SPLITS.map((s) =>
            counts[s] > 0 ? <div key={s} style={{ flex: counts[s], background: SPLIT_COLORS[s] }} /> : null,
          )}
        </div>
      </div>
    </Tooltip>
  )
}

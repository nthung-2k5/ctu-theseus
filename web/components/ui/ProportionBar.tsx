import { Tooltip } from '@mantine/core'

export interface ProportionSegment {
  key: string
  label: string
  value: number
  color: string
  /** Hatched fill, for augmented shares (colour is never the only channel). */
  dashed?: boolean
}

/** Stacked bar showing how a total divides into coloured segments (e.g. train/validation/test). */
export function ProportionBar({ segments, height = 12 }: { segments: ProportionSegment[]; height?: number }) {
  const total = segments.reduce((sum, s) => sum + s.value, 0)
  return (
    <div
      style={{
        display: 'flex',
        width: '100%',
        height,
        borderRadius: height / 2,
        overflow: 'hidden',
        background: 'var(--mantine-color-dark-6)',
      }}
    >
      {total > 0 &&
        segments
          .filter((s) => s.value > 0)
          .map((s) => (
            <Tooltip key={s.key} label={`${s.label}: ${s.value}`}>
              <div
                style={{
                  width: `${(s.value / total) * 100}%`,
                  background: s.dashed
                    ? `repeating-linear-gradient(45deg, ${s.color} 0 5px, transparent 5px 8px)`
                    : s.color,
                }}
              />
            </Tooltip>
          ))}
    </div>
  )
}

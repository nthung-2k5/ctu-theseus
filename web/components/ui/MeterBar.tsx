/** Thin horizontal bar on a dark track, used for per-class counts and confidence values. */
export function MeterBar({
  value,
  max = 100,
  color = 'var(--mantine-color-cyan-5)',
  height = 6,
}: {
  value: number
  max?: number
  color?: string
  height?: number
}) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0
  return (
    // biome-ignore lint/a11y/useSemanticElements: a styled bar; <meter> cannot be themed consistently across browsers
    <div
      role="meter"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      style={{
        width: '100%',
        height,
        borderRadius: height / 2,
        background: 'var(--mantine-color-dark-6)',
        overflow: 'hidden',
      }}
    >
      <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width 120ms ease' }} />
    </div>
  )
}

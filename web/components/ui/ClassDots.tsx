export interface ClassDotSource {
  id: string
  name: string
  color?: string | null
}

const FALLBACK = '#64748b'

export function ClassDot({ color, size = 10 }: { color?: string | null; size?: number }) {
  return (
    <span
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        borderRadius: '50%',
        background: color ?? FALLBACK,
        flex: 'none',
      }}
    />
  )
}

/** Overlapping dots that preview a class list on cards. */
export function ClassDots({ classes, max = 10 }: { classes: readonly ClassDotSource[]; max?: number }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center' }} role="img" aria-label={`${classes.length} classes`}>
      {classes.slice(0, max).map((c, i) => (
        <span
          key={c.id}
          title={c.name}
          style={{
            width: 12,
            height: 12,
            borderRadius: '50%',
            background: c.color ?? FALLBACK,
            marginLeft: i ? -3 : 0,
            border: '2px solid var(--mantine-color-body)',
          }}
        />
      ))}
      {classes.length > max && (
        <span style={{ marginLeft: 6, fontSize: 11, opacity: 0.7 }}>+{classes.length - max}</span>
      )}
    </span>
  )
}

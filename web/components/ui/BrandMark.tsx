/** The gradient square next to the product name. */
export function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <div
      aria-hidden
      style={{
        width: size,
        height: size,
        borderRadius: Math.round(size * 0.27),
        background: 'linear-gradient(135deg,#1f5ca9,#00afef)',
        flex: 'none',
      }}
    />
  )
}

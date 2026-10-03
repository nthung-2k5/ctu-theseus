import { Badge, type BadgeProps } from '@mantine/core'

/**
 * A badge colored by looking `value` up in a color map — the same pattern
 * repeated for training status, dataset modality, and split type, each with
 * its own map (see lib/constants.ts, components/training/constants.ts).
 */
export function StatusBadge({
  value,
  colorMap,
  fallbackColor = 'gray',
  ...badgeProps
}: {
  value: string
  colorMap: Record<string, string>
  fallbackColor?: string
} & Omit<BadgeProps, 'color' | 'children'>) {
  const color = colorMap[value] ?? fallbackColor
  // A raw hex colour (the chart palette) is used as the light variant's text colour as-is, which
  // is too pale on its tint in the light scheme. Darken it there; dark keeps the bright colour.
  const style = color.startsWith('#')
    ? ({ '--badge-color': `light-dark(color-mix(in srgb, ${color}, black 50%), ${color})` } as React.CSSProperties)
    : undefined
  return (
    <Badge variant="light" color={color} tt="capitalize" style={style} {...badgeProps}>
      {value.replace(/_/g, ' ')}
    </Badge>
  )
}

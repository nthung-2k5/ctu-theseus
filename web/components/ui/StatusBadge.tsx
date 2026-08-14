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
  return (
    <Badge variant="light" color={colorMap[value] ?? fallbackColor} tt="capitalize" {...badgeProps}>
      {value.replace(/_/g, ' ')}
    </Badge>
  )
}

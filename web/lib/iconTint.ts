import type { CSSProperties } from 'react'

/**
 * Style for a `variant="light"` ThemeIcon whose square stays visible against the card behind it.
 *
 * Mantine's stock light tint is the colour's palest shade (light mode) or a 15%-alpha dark shade (dark mode), which
 * all but disappears on a Paper, grey most of all. This mixes the variant's own foreground colour into the
 * background instead: that colour is light in dark mode and dark in light mode, so the square lifts off the card in
 * both. A matching hairline border defines the edge.
 */
export const strongIconTint = (color: string): CSSProperties =>
  ({
    '--ti-bg': `color-mix(in srgb, var(--mantine-color-${color}-light-color) 16%, transparent)`,
    '--ti-bd': `color-mix(in srgb, var(--mantine-color-${color}-light-color) 38%, transparent)`,
  }) as CSSProperties

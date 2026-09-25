/**
 * Semantic colours shared by charts, badges and bars. Values are plain CSS colours
 * (Mantine's `color` props accept them), chosen to stay legible on the slate `dark`
 * surface ramp defined in theme.ts.
 */

/** Series colours for metric charts. Validation is additionally dashed (colour is never the only channel). */
export const SERIES_COLORS = {
  train: '#00afef',
  validation: '#f59e0b',
  f1: '#34d399',
  learningRate: '#a78bfa',
} as const

/** Dash pattern for validation series (recharts `strokeDasharray`). */
export const VALIDATION_DASH = '5 4'

/** Dataset partition colours. */
export const PARTITION_COLORS = {
  train: '#4a89d1',
  validation: '#00afef',
  test: '#f59e0b',
} as const

/** Accent for augmented items/charts. */
export const AUGMENTED_COLOR = '#a78bfa'

/** Ordered categorical palette for multi-series charts. */
export const CHART_COLORS = ['#00afef', '#f59e0b', '#34d399', '#f472b6', '#a78bfa', '#4a89d1']

/** Confusion-matrix / heatmap ramp, low to high. */
export const HEATMAP_RAMP = ['#0f172a', '#1f5ca9', '#00afef', '#e0f7ff']

const hex = (c: string) => [1, 3, 5].map((i) => Number.parseInt(c.slice(i, i + 2), 16))

/** Colour at position `t` (0-1) along HEATMAP_RAMP, linearly interpolated in RGB. */
export function heatmapColor(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (HEATMAP_RAMP.length - 1)
  const i = Math.min(HEATMAP_RAMP.length - 2, Math.floor(x))
  const a = hex(HEATMAP_RAMP[i])
  const b = hex(HEATMAP_RAMP[i + 1])
  const f = x - i
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`
}

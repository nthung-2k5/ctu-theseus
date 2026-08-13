import { createTheme, type MantineThemeOverride } from '@mantine/core'

/**
 * Custom Mantine theme.
 *
 * Primary:   #1f5ca9
 * Secondary: #00afef
 * Title font: K2D
 * Body font:  Readex Pro
 */

const PRIMARY_SHADES: [string, string, string, string, string, string, string, string, string, string] = [
  '#e8f0fb',
  '#c5d9f3',
  '#9fbfea',
  '#79a4e0',
  '#5389d5',
  '#1f5ca9',
  '#1a4e91',
  '#154079',
  '#103261',
  '#0b244a',
]

const SECONDARY_SHADES: [string, string, string, string, string, string, string, string, string, string] = [
  '#e0f6fd',
  '#b3e8fa',
  '#80d9f6',
  '#4dcaf2',
  '#1abbee',
  '#00afef',
  '#0094cb',
  '#0079a7',
  '#005e83',
  '#00435f',
]

export const theme: MantineThemeOverride = createTheme({
  primaryColor: 'primary',
  fontFamily: "'Readex Pro', sans-serif",
  headings: {
    fontFamily: "'K2D', sans-serif",
    fontWeight: '600',
  },
  colors: {
    primary: PRIMARY_SHADES,
    secondary: SECONDARY_SHADES,
  },
  defaultRadius: 'md',
  cursorType: 'pointer',
  other: {
    primaryHex: '#1f5ca9',
    secondaryHex: '#00afef',
  },
})

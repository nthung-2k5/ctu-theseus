import { createTheme, type MantineColorsTuple, type MantineThemeOverride, rem } from '@mantine/core'

/**
 * Custom Mantine theme (dark-first, dense, desktop-first).
 *
 * Primary:   #1f5ca9 (deep academic blue, shade 7 in light / 5 in dark)
 * Secondary: #00afef (vibrant cyan, used for text/lines that must stand out on dark)
 * Font:      Readex Pro (UI), JetBrains Mono (code, logs, numbers)
 *
 * The palette names `primary` and `secondary` are kept so existing `color="primary"`
 * props and the Tailwind `@theme` aliases in global.css keep working.
 */

const PRIMARY_SHADES: MantineColorsTuple = [
  '#e8f1fb',
  '#cfe0f4',
  '#a1c3e8',
  '#6fa3dc',
  '#4a89d1',
  '#2f74c6',
  '#256bbb',
  '#1f5ca9',
  '#1a4f92',
  '#12396d',
]

const SECONDARY_SHADES: MantineColorsTuple = [
  '#e0f7ff',
  '#b8ebff',
  '#8adeff',
  '#52cfff',
  '#22c1fb',
  '#00afef',
  '#0099d1',
  '#007fae',
  '#00668b',
  '#004a66',
]

/** Slate surface ramp: dark[0] = text, dark[9] = deepest background. */
const DARK_SHADES: MantineColorsTuple = [
  '#e2e8f0',
  '#cbd5e1',
  '#94a3b8',
  '#7c8aa0',
  '#475569',
  '#334155',
  '#1e293b',
  '#172033',
  '#0f172a',
  '#0a0f1c',
]

export const MONO_STACK = "'JetBrains Mono', 'Fira Code', ui-monospace, monospace"

export const theme: MantineThemeOverride = createTheme({
  primaryColor: 'primary',
  primaryShade: { light: 7, dark: 5 },
  colors: {
    primary: PRIMARY_SHADES,
    secondary: SECONDARY_SHADES,
    cyan: SECONDARY_SHADES,
    dark: DARK_SHADES,
  },
  fontFamily: "'Readex Pro', sans-serif",
  fontFamilyMonospace: MONO_STACK,
  headings: { fontFamily: "'Readex Pro', sans-serif", fontWeight: '500' },
  defaultRadius: 'sm',
  cursorType: 'pointer',
  black: '#0a0f1c',
  white: '#f8fafc',
  fontSizes: { xs: rem(11), sm: rem(12.5), md: rem(14), lg: rem(16), xl: rem(18) },
  spacing: { xs: rem(6), sm: rem(10), md: rem(14), lg: rem(20), xl: rem(28) },
  other: {
    primaryHex: '#1f5ca9',
    secondaryHex: '#00afef',
    chartPalette: ['#00afef', '#4a89d1', '#f59e0b', '#34d399', '#f472b6', '#a78bfa'],
  },
  components: {
    Button: { defaultProps: { size: 'compact-md' } },
    Badge: { defaultProps: { radius: 'sm', variant: 'light' } },
    Paper: { defaultProps: { withBorder: true, radius: 'md' } },
    Tabs: { defaultProps: { variant: 'default' } },
    SegmentedControl: { defaultProps: { size: 'xs' } },
    Tooltip: { defaultProps: { withArrow: true, openDelay: 250 } },
    Code: { styles: { root: { fontFamily: MONO_STACK } } },
  },
})

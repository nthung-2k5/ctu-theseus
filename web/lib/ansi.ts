/** Minimal ANSI SGR parser: colours (30-37, 90-97, 38;5;n, 38;2;r;g;b), bold, dim, reset. */

export interface AnsiSpan {
  text: string
  color?: string
  bold?: boolean
  dim?: boolean
}

const BASE = ['#0f172a', '#f87171', '#4ade80', '#fbbf24', '#60a5fa', '#c084fc', '#22d3ee', '#e2e8f0']
const BRIGHT = ['#64748b', '#fca5a5', '#86efac', '#fde68a', '#93c5fd', '#d8b4fe', '#67e8f9', '#f8fafc']

function xterm256(n: number): string {
  if (n < 8) return BASE[n]!
  if (n < 16) return BRIGHT[n - 8]!
  if (n < 232) {
    const c = n - 16
    const v = (k: number) => (k === 0 ? 0 : 55 + k * 40)
    return `rgb(${v(Math.floor(c / 36))},${v(Math.floor(c / 6) % 6)},${v(c % 6)})`
  }
  const g = 8 + (n - 232) * 10
  return `rgb(${g},${g},${g})`
}

// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC (0x1b) introduces every ANSI sequence
const SGR = /\u001b\[([0-9;]*)m/g
// Any other CSI sequence (cursor movement, erase) is dropped rather than rendered as garbage.
// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC (0x1b) introduces every ANSI sequence
const OTHER_CSI = /\u001b\[[0-9;?]*[A-LN-Za-ln-z]/g

const cache = new Map<string, AnsiSpan[]>()

export function stripAnsi(s: string): string {
  return s.replace(SGR, '').replace(OTHER_CSI, '')
}

export function parseAnsi(input: string): AnsiSpan[] {
  const hit = cache.get(input)
  if (hit) return hit
  const text = input.replace(OTHER_CSI, '')
  const spans: AnsiSpan[] = []
  let color: string | undefined
  let bold = false
  let dim = false
  let last = 0

  const push = (chunk: string) => {
    if (chunk) spans.push({ text: chunk, color, bold: bold || undefined, dim: dim || undefined })
  }

  for (const m of text.matchAll(SGR)) {
    push(text.slice(last, m.index))
    last = (m.index ?? 0) + m[0].length
    const codes = (m[1] === '' ? '0' : m[1]!).split(';').map(Number)
    for (let i = 0; i < codes.length; i++) {
      const c = codes[i]!
      if (c === 0) {
        color = undefined
        bold = false
        dim = false
      } else if (c === 1) bold = true
      else if (c === 2) dim = true
      else if (c === 22) {
        bold = false
        dim = false
      } else if (c === 39) color = undefined
      else if (c >= 30 && c <= 37) color = BASE[c - 30]
      else if (c >= 90 && c <= 97) color = BRIGHT[c - 90]
      else if (c === 38 && codes[i + 1] === 5) {
        color = xterm256(codes[i + 2] ?? 7)
        i += 2
      } else if (c === 38 && codes[i + 1] === 2) {
        color = `rgb(${codes[i + 2] ?? 0},${codes[i + 3] ?? 0},${codes[i + 4] ?? 0})`
        i += 4
      }
    }
  }
  push(text.slice(last))

  if (cache.size > 4000) cache.clear()
  cache.set(input, spans)
  return spans
}

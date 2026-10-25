import { expect, test } from 'bun:test'
import { formatRelative } from './format'

const now = Date.parse('2026-06-15T12:00:00Z')
const ago = (ms: number) => new Date(now - ms).toISOString()

test('relative time picks the unit that reads best', () => {
  expect(formatRelative(ago(20_000), now)).toBe('just now')
  expect(formatRelative(ago(5 * 60_000), now)).toBe('5m ago')
  expect(formatRelative(ago(3 * 3_600_000), now)).toBe('3h ago')
  expect(formatRelative(ago(2 * 86_400_000), now)).toBe('2d ago')
})

test('a month or more falls back to a date, and missing input to a dash', () => {
  expect(formatRelative(ago(45 * 86_400_000), now)).not.toContain('ago')
  expect(formatRelative(null, now)).toBe('—')
  expect(formatRelative('not a date', now)).toBe('—')
})

test('a time slightly in the future (clock skew) reads as just now, not a negative age', () => {
  expect(formatRelative(ago(-30_000), now)).toBe('just now')
})

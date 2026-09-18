import { describe, expect, test } from 'bun:test'
import { checkRateLimit } from './rateLimit'

describe('checkRateLimit', () => {
  test('allows up to maxRequests within the window, then blocks', () => {
    const key = `test-key-${crypto.randomUUID()}`
    expect(checkRateLimit(key, 3, 60_000)).toBe(true)
    expect(checkRateLimit(key, 3, 60_000)).toBe(true)
    expect(checkRateLimit(key, 3, 60_000)).toBe(true)
    expect(checkRateLimit(key, 3, 60_000)).toBe(false)
  })

  test('different keys have independent windows', () => {
    const keyA = `test-key-a-${crypto.randomUUID()}`
    const keyB = `test-key-b-${crypto.randomUUID()}`
    expect(checkRateLimit(keyA, 1, 60_000)).toBe(true)
    expect(checkRateLimit(keyA, 1, 60_000)).toBe(false)
    // keyB has never been seen, so it gets its own fresh allowance.
    expect(checkRateLimit(keyB, 1, 60_000)).toBe(true)
  })

  test('resets once the window elapses', async () => {
    const key = `test-key-${crypto.randomUUID()}`
    expect(checkRateLimit(key, 1, 20)).toBe(true)
    expect(checkRateLimit(key, 1, 20)).toBe(false)
    await Bun.sleep(30)
    expect(checkRateLimit(key, 1, 20)).toBe(true)
  })
})

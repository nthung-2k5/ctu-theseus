import { describe, expect, test } from 'bun:test'
import { generateApiKey, hashApiKey } from './apiKeys'

describe('generateApiKey', () => {
  test('the raw key carries a recognizable prefix', () => {
    const { rawKey } = generateApiKey()
    expect(rawKey.startsWith('thsk_')).toBe(true)
  })

  test('keyHash is the sha256 of the raw key, reproducible via hashApiKey', () => {
    const { rawKey, keyHash } = generateApiKey()
    expect(hashApiKey(rawKey)).toBe(keyHash)
  })

  test('keyPrefix is a genuine prefix of the raw key, short enough to not be useful to an attacker', () => {
    const { rawKey, keyPrefix } = generateApiKey()
    expect(rawKey.startsWith(keyPrefix)).toBe(true)
    expect(keyPrefix.length).toBeLessThan(rawKey.length / 2)
  })

  test('two calls never produce the same key', () => {
    const a = generateApiKey()
    const b = generateApiKey()
    expect(a.rawKey).not.toBe(b.rawKey)
    expect(a.keyHash).not.toBe(b.keyHash)
  })
})

describe('hashApiKey', () => {
  test('is deterministic', () => {
    expect(hashApiKey('thsk_same')).toBe(hashApiKey('thsk_same'))
  })

  test('different input produces a different hash', () => {
    expect(hashApiKey('thsk_a')).not.toBe(hashApiKey('thsk_b'))
  })
})

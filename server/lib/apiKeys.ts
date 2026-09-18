/**
 * API key generation/hashing for the hosted prediction API (see
 * routes/api-v1.ts and the `apiKeyAuth` macro in routes/auth.ts).
 *
 * Only a sha256 hash of a key is ever persisted (`apiKeys.keyHash` in
 * db/schema.ts) — the raw key is returned to the caller exactly once, at
 * creation, and is unrecoverable after that (same principle as a password).
 */

const KEY_PREFIX = 'thsk_'
const RAW_KEY_BYTES = 24
/** How much of the raw key is safe to keep around for display — enough to tell keys apart, not enough to be useful to an attacker. */
const DISPLAY_PREFIX_LENGTH = KEY_PREFIX.length + 6

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export function hashApiKey(rawKey: string): string {
  return new Bun.CryptoHasher('sha256').update(rawKey).digest('hex')
}

/** Generates a new raw key plus everything needed to persist it — the raw key itself is not returned by any other function in this module. */
export function generateApiKey(): { rawKey: string; keyHash: string; keyPrefix: string } {
  const rawKey = KEY_PREFIX + toHex(crypto.getRandomValues(new Uint8Array(RAW_KEY_BYTES)))
  return {
    rawKey,
    keyHash: hashApiKey(rawKey),
    keyPrefix: rawKey.slice(0, DISPLAY_PREFIX_LENGTH),
  }
}

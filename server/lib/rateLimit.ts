/**
 * Minimal in-memory fixed-window rate limiter — used by the hosted
 * prediction API's `apiKeyAuth` macro (routes/auth.ts), the first route in
 * this app not behind a session cookie and therefore the first that needs
 * one at all.
 *
 * Per-process state: correct for a single gateway instance, which is the
 * only topology apphost.mts actually provisions today (no replica count is
 * configured for the gateway resource). Scaling the gateway out would need
 * this moved to a shared store (a NATS KV bucket, or Redis) — each replica
 * would otherwise enforce its own independent limit, silently multiplying
 * the effective allowance by the replica count.
 */

interface Window {
  count: number
  windowStartMs: number
}

const windows = new Map<string, Window>()

/**
 * Returns whether `key` is still within `maxRequests` for the current
 * `windowMs`-long window, incrementing its count as a side effect. `key`
 * should never be a raw secret (an API key's hash, not the key itself) —
 * this map lives for the life of the process.
 */
export function checkRateLimit(key: string, maxRequests: number, windowMs: number): boolean {
  const now = Date.now()
  const existing = windows.get(key)

  if (!existing || now - existing.windowStartMs >= windowMs) {
    windows.set(key, { count: 1, windowStartMs: now })
    return true
  }
  if (existing.count >= maxRequests) return false
  existing.count++
  return true
}

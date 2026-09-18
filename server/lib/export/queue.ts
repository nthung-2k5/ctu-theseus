/**
 * In-process, concurrency-limited runner for bundle assembly jobs.
 *
 * Assembly must never run inline in the NATS event handler (see the
 * `export` case in lib/microservice.ts) — zipping a large model can exceed
 * the consumer's ack_wait and trigger a duplicate redelivery. This queue is
 * the buffer between "an export became assemblable" (an event, an API
 * dispatch, or the lazy-reconcile check in routes/export.ts) and the actual
 * work.
 */

import { buildBundle } from './bundle'

const MAX_CONCURRENCY = 2

let active = 0
const pending: string[] = []
const queued = new Set<string>()

export function enqueueAssembly(exportId: string): void {
  if (queued.has(exportId)) return
  queued.add(exportId)
  pending.push(exportId)
  pump()
}

function pump(): void {
  while (active < MAX_CONCURRENCY && pending.length > 0) {
    const exportId = pending.shift()
    if (!exportId) break
    queued.delete(exportId)
    active++
    buildBundle(exportId)
      .catch((e) => console.error(`[export] unhandled error assembling ${exportId}:`, e))
      .finally(() => {
        active--
        pump()
      })
  }
}

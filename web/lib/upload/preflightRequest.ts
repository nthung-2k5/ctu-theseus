/** The server half of the preflight: which of these content digests does the draft already hold? */

import { preflightItems } from '@public/lib/api/generated/datasets/datasets'

/** Digests per request; the server accepts 5000. */
const HASHES_PER_REQUEST = 2000

export async function findExisting(projectId: string, hashes: string[], signal?: AbortSignal): Promise<Set<string>> {
  const existing = new Set<string>()
  const unique = [...new Set(hashes)]
  for (let start = 0; start < unique.length; start += HASHES_PER_REQUEST) {
    signal?.throwIfAborted()
    const response = await preflightItems(
      projectId,
      { hashes: unique.slice(start, start + HASHES_PER_REQUEST) },
      undefined,
      signal,
    )
    for (const hash of response.existing) existing.add(hash)
  }
  return existing
}

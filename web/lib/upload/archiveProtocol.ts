/** Messages and limits shared by the archive worker and its main-thread wrapper. */

import type { ArchiveFormat } from './paths'

/** More entries than this and the archive is rejected outright. */
export const MAX_ARCHIVE_ENTRIES = 50_000

/**
 * Cap on *uncompressed* bytes read out of one archive, counting entries that were then discarded. Extracted
 * files live in browser memory, so this is what stands between a zip bomb and a crashed tab.
 */
export const MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024

export interface ExtractRequest {
  file: File
  format: ArchiveFormat
}

export interface WorkerEntry {
  path: string[]
  bytes: ArrayBuffer
}

export type WorkerMessage =
  | { type: 'entries'; items: WorkerEntry[] }
  | { type: 'skipped'; path: string; reason: string }
  | { type: 'progress'; read: number; total: number }
  | { type: 'done'; junk: number }
  | { type: 'error'; message: string }

/**
 * The check that runs between staging and uploading: read every file once, work out which ones need not be sent,
 * and say why.
 *
 * A file is skipped when the draft already holds its content (the server says so, see `preflightItems`) or when an
 * earlier staged file has the same bytes. Both are exactly the cases where sending would change nothing: the server
 * stores content once per project, keeps the first split and never relabels a hit. So the saving costs no behaviour.
 * The whole check is advisory: a file it could not read or ask about is simply sent, and the server dedups it again.
 */

export const IN_DATASET_REASON = 'Already in this dataset'
export const SAME_AS_PREFIX = 'Same content as '

/** Bytes read into memory at once while hashing, and the most files hashed side by side. */
const MAX_IN_FLIGHT_BYTES = 96 * 1024 * 1024
const MAX_WORKERS = 4

/** `crypto.subtle` only exists in a secure context; elsewhere the check is skipped, never faked. */
export const canHash = (): boolean => typeof globalThis.crypto?.subtle?.digest === 'function'

export async function sha256Hex(file: Blob): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Lets callers reserve bytes and wait while too many are already held. A request bigger than the budget runs alone. */
class ByteBudget {
  private held = 0
  private waiting: (() => void)[] = []
  private readonly limit: number

  constructor(limit: number) {
    this.limit = limit
  }

  async acquire(bytes: number): Promise<void> {
    const want = Math.min(bytes, this.limit)
    while (this.held > 0 && this.held + want > this.limit) {
      await new Promise<void>((resolve) => this.waiting.push(resolve))
    }
    this.held += want
  }

  release(bytes: number): void {
    this.held -= Math.min(bytes, this.limit)
    for (const wake of this.waiting.splice(0)) wake()
  }
}

export interface HashRequest {
  id: string
  file: File
}

/**
 * Hashes the files with bounded memory and concurrency, reporting each digest as soon as it is known. A file that
 * cannot be read (it vanished from disk, say) is left out rather than failing the rest.
 */
export async function hashFiles(
  files: HashRequest[],
  options: {
    signal?: AbortSignal
    onHashed?: (id: string, hash: string) => void
    /** 0..1 by bytes, so one huge video does not read as "one file of many". */
    onProgress?: (fraction: number) => void
  } = {},
): Promise<Map<string, string>> {
  const { signal, onHashed, onProgress } = options
  const hashes = new Map<string, string>()
  const totalBytes = files.reduce((n, f) => n + f.file.size, 0) || 1
  const budget = new ByteBudget(MAX_IN_FLIGHT_BYTES)
  let doneBytes = 0
  let next = 0

  const worker = async () => {
    for (;;) {
      signal?.throwIfAborted()
      const entry = files[next++]
      if (!entry) return
      await budget.acquire(entry.file.size)
      try {
        signal?.throwIfAborted()
        const hash = await sha256Hex(entry.file)
        hashes.set(entry.id, hash)
        onHashed?.(entry.id, hash)
      } catch (error) {
        if (signal?.aborted) throw error
        // unreadable: leave it unhashed, so it is sent and the server decides
      } finally {
        budget.release(entry.file.size)
        doneBytes += entry.file.size
        onProgress?.(Math.min(1, doneBytes / totalBytes))
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(MAX_WORKERS, files.length) }, worker))
  return hashes
}

export interface Hashed {
  id: string
  name: string
  hash?: string
}

/**
 * Which files to skip, and why: `id -> reason`. Files are read in staging order, so the first copy of a content is
 * the one that is kept and later copies point back at it.
 */
export function planSkips(files: Iterable<Hashed>, existing: ReadonlySet<string>): Record<string, string> {
  const reasons: Record<string, string> = {}
  const firstByHash = new Map<string, string>()
  for (const { id, name, hash } of files) {
    if (!hash) continue
    if (existing.has(hash)) {
      reasons[id] = IN_DATASET_REASON
      continue
    }
    const first = firstByHash.get(hash)
    if (first === undefined) firstByHash.set(hash, name)
    else reasons[id] = `${SAME_AS_PREFIX}${first}`
  }
  return reasons
}

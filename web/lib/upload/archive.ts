/** Main-thread side of archive extraction: runs `archiveWorker.ts` and turns its output into `File`s. */

import type { WorkerMessage } from './archiveProtocol'
import { type ArchiveFormat, mimeFromName } from './paths'
import type { RawEntry } from './types'

export interface Skipped {
  path: string
  reason: string
}

export interface ExtractResult {
  entries: RawEntry[]
  skipped: Skipped[]
  /** OS metadata entries (`__MACOSX/`, `.DS_Store`...) dropped inside the worker. */
  junk: number
}

/**
 * Extracts `file` in a worker. Rejects when the archive is unreadable or breaks a limit (nothing from a
 * rejected archive is returned), and with an `AbortError` when `signal` fires.
 */
export function extractArchive(
  file: File,
  format: ArchiveFormat,
  options: { signal?: AbortSignal; onProgress?: (fraction: number) => void } = {},
): Promise<ExtractResult> {
  const { signal, onProgress } = options
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }
    const worker = new Worker(new URL('./archiveWorker.ts', import.meta.url), { type: 'module' })
    const entries: RawEntry[] = []
    const skipped: Skipped[] = []

    const finish = (settle: () => void) => {
      signal?.removeEventListener('abort', onAbort)
      worker.terminate()
      settle()
    }
    const onAbort = () => finish(() => reject(new DOMException('Aborted', 'AbortError')))
    signal?.addEventListener('abort', onAbort, { once: true })

    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data
      switch (message.type) {
        case 'entries':
          for (const { path, bytes } of message.items) {
            const name = path[path.length - 1]
            entries.push({ path, file: new File([bytes], name, { type: mimeFromName(name) }) })
          }
          break
        case 'skipped':
          skipped.push({ path: message.path, reason: message.reason })
          break
        case 'progress':
          if (message.total > 0) onProgress?.(message.read / message.total)
          break
        case 'done':
          finish(() => resolve({ entries, skipped, junk: message.junk }))
          break
        case 'error':
          finish(() => reject(new Error(message.message)))
          break
      }
    }
    worker.onerror = () => finish(() => reject(new Error('The archive could not be read')))
    worker.postMessage({ file, format })
  })
}

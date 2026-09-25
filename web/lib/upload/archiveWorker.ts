/**
 * Extracts a zip / tar / gzip archive off the main thread, streaming: the archive is read in chunks and
 * entries are posted back in batches, so a large archive never sits fully decompressed in one buffer.
 *
 * Guards (see `archiveProtocol.ts` for the limits): unsafe paths, OS junk, entries over the 50 MB upload
 * limit, and a cap on entry count and total inflated bytes are all handled here, before anything is handed
 * to the page.
 */

import { Gunzip, Unzip, UnzipInflate, UnzipPassThrough } from 'fflate'
import {
  type ExtractRequest,
  MAX_ARCHIVE_BYTES,
  MAX_ARCHIVE_ENTRIES,
  type WorkerEntry,
  type WorkerMessage,
} from './archiveProtocol'
import { isJunkPath, looksLikeTar, MAX_FILE_BYTES, normalizePath } from './paths'
import { TarParser } from './tar'

const scope = self as unknown as {
  postMessage(message: WorkerMessage, transfer?: Transferable[]): void
  onmessage: ((event: MessageEvent<ExtractRequest>) => void) | null
}

const post = (message: WorkerMessage, transfer?: Transferable[]) => scope.postMessage(message, transfer)

/** Aborts the whole archive (as opposed to skipping one entry); its message is shown to the user. */
class LimitError extends Error {}

const SIZE_LIMIT_REASON = `Larger than the ${MAX_FILE_BYTES / 1024 / 1024} MB limit`

function concat(chunks: Uint8Array[], size: number): Uint8Array {
  const out = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

/** Gathers one entry at a time, applies the guards, and posts finished entries in batches. */
class Collector {
  junk = 0
  private entries = 0
  private inflated = 0
  private batch: WorkerEntry[] = []
  private batchBytes = 0
  private current: { path: string[]; chunks: Uint8Array[]; size: number } | null = null

  /** Starts an entry. `false` = don't feed it data (unsafe, junk, too big). */
  begin(rawPath: string, declaredSize?: number): boolean {
    const path = normalizePath(rawPath)
    if (!path.ok) {
      post({ type: 'skipped', path: rawPath, reason: path.reason })
      return false
    }
    if (isJunkPath(path.segments)) {
      this.junk++
      return false
    }
    if (declaredSize !== undefined && declaredSize > MAX_FILE_BYTES) {
      post({ type: 'skipped', path: rawPath, reason: SIZE_LIMIT_REASON })
      return false
    }
    if (this.entries >= MAX_ARCHIVE_ENTRIES) {
      throw new LimitError(`Archive has more than ${MAX_ARCHIVE_ENTRIES.toLocaleString('en-US')} files`)
    }
    this.current = { path: path.segments, chunks: [], size: 0 }
    return true
  }

  /** `false` = the entry turned out too big while inflating and was dropped; stop feeding it. */
  data(chunk: Uint8Array): boolean {
    const entry = this.current
    if (!entry) return false
    this.inflated += chunk.length
    if (this.inflated > MAX_ARCHIVE_BYTES) {
      throw new LimitError(`Archive expands to more than ${MAX_ARCHIVE_BYTES / 1024 / 1024 / 1024} GB`)
    }
    entry.size += chunk.length
    if (entry.size > MAX_FILE_BYTES) {
      post({ type: 'skipped', path: entry.path.join('/'), reason: SIZE_LIMIT_REASON })
      this.current = null
      return false
    }
    entry.chunks.push(chunk)
    return true
  }

  /** Drops the current entry with a reason (corrupt, encrypted...). */
  abort(reason: string): void {
    if (this.current) post({ type: 'skipped', path: this.current.path.join('/'), reason })
    this.current = null
  }

  end(): void {
    const entry = this.current
    if (!entry) return
    this.current = null
    const bytes = concat(entry.chunks, entry.size)
    this.entries++
    this.batch.push({ path: entry.path, bytes: bytes.buffer as ArrayBuffer })
    this.batchBytes += entry.size
    if (this.batch.length >= 64 || this.batchBytes >= 16 * 1024 * 1024) this.flush()
  }

  flush(): void {
    if (this.batch.length === 0) return
    post(
      { type: 'entries', items: this.batch },
      this.batch.map((item) => item.bytes),
    )
    this.batch = []
    this.batchBytes = 0
  }
}

interface Feeder {
  push(chunk: Uint8Array, final: boolean): void
}

function zipFeeder(collector: Collector): Feeder {
  const unzip = new Unzip()
  unzip.register(UnzipInflate)
  unzip.register(UnzipPassThrough)
  unzip.onfile = (file) => {
    if (file.name.endsWith('/')) return // a directory: never started, so fflate skips it
    if (!collector.begin(file.name, file.originalSize)) return
    let dead = false
    file.ondata = (err, chunk, final) => {
      if (dead) return
      if (err) {
        dead = true
        collector.abort('Could not be read (corrupt or password-protected)')
        file.terminate()
        return
      }
      if (!collector.data(chunk)) {
        dead = true
        file.terminate()
        return
      }
      if (final) collector.end()
    }
    file.start()
  }
  return { push: (chunk, final) => unzip.push(chunk, final) }
}

function tarFeeder(collector: Collector): Feeder {
  const parser = new TarParser({
    onFile: (path, size) => collector.begin(path, size),
    // The parser reuses its input chunks' memory, so keep our own copy.
    onData: (chunk) => void collector.data(chunk.slice()),
    onFileEnd: () => collector.end(),
    onSkipped: (path, reason) => post({ type: 'skipped', path, reason }),
  })
  return {
    push: (chunk, final) => {
      parser.push(chunk)
      if (final) parser.end()
    },
  }
}

/** `.tar.gz` streams into the tar reader; a lone `photo.png.gz` becomes a single file. */
function gzipFeeder(collector: Collector, archiveName: string): Feeder {
  const gunzip = new Gunzip()
  let tar: Feeder | null = null
  let single: 'live' | 'dead' | null = null
  let head: Uint8Array = new Uint8Array(0)
  let decided = false

  const sink = (input: Uint8Array, final: boolean) => {
    let chunk = input
    if (!decided) {
      const merged = new Uint8Array(head.length + chunk.length)
      merged.set(head)
      merged.set(chunk, head.length)
      head = merged
      if (head.length < 512 && !final) return
      decided = true
      if (looksLikeTar(head)) tar = tarFeeder(collector)
      else single = collector.begin(archiveName.replace(/\.gz$/i, ''), undefined) ? 'live' : 'dead'
      chunk = head
      head = new Uint8Array(0)
    }
    if (tar) {
      tar.push(chunk, final)
    } else if (single === 'live') {
      if (!collector.data(chunk)) single = 'dead'
      else if (final) collector.end()
    }
  }

  gunzip.ondata = (chunk, final) => sink(chunk, final)
  return { push: (chunk, final) => gunzip.push(chunk, final) }
}

async function extract({ file, format }: ExtractRequest): Promise<void> {
  const collector = new Collector()
  const feeder =
    format === 'zip' ? zipFeeder(collector) : format === 'tar' ? tarFeeder(collector) : gzipFeeder(collector, file.name)

  const reader = file.stream().getReader()
  let read = 0
  let lastProgress = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    read += value.length
    feeder.push(value, false)
    const now = Date.now()
    if (now - lastProgress > 100) {
      lastProgress = now
      post({ type: 'progress', read, total: file.size })
    }
  }
  feeder.push(new Uint8Array(0), true)
  collector.flush()
  post({ type: 'progress', read: file.size, total: file.size })
  post({ type: 'done', junk: collector.junk })
}

scope.onmessage = (event) => {
  extract(event.data).catch((error: unknown) => {
    post({ type: 'error', message: error instanceof Error ? error.message : 'The archive could not be read' })
  })
}

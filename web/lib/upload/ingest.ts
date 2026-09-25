/**
 * Turns whatever the user handed us — a drop of files/folders/archives, or a file/folder picker result —
 * into a flat list of `RawEntry` (folder path + `File`). Archives are unpacked here (in a worker); junk and
 * unsafe paths are filtered out and counted so the UI can say what it left behind.
 */

import { extractArchive, type Skipped } from './archive'
import { archiveFormatFromName, isJunkPath, normalizePath, sniffArchive } from './paths'
import type { RawEntry } from './types'

/** A file plus the path the browser reported for it (`/pets/cat/a.jpg`, `cat/a.jpg` or just `a.jpg`). */
export interface SourceFile {
  file: File
  path: string
}

export interface IngestProgress {
  label: string
  /** 0..1 */
  fraction: number
}

export interface IngestResult {
  entries: RawEntry[]
  skipped: Skipped[]
  junk: number
}

export interface IngestOptions {
  signal?: AbortSignal
  onProgress?: (progress: IngestProgress) => void
}

/** Folder-picker and `<input multiple>` results. `webkitRelativePath` is empty for plain file picks. */
export function sourcesFromFiles(files: Iterable<File>): SourceFile[] {
  return Array.from(files, (file) => ({ file, path: file.webkitRelativePath || file.name }))
}

function readFile(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => entry.file(resolve, reject))
}

function readBatch(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject))
}

async function walk(entry: FileSystemEntry, sources: SourceFile[], failed: Skipped[]): Promise<void> {
  try {
    if (entry.isFile) {
      sources.push({ file: await readFile(entry as FileSystemFileEntry), path: entry.fullPath })
      return
    }
    if (!entry.isDirectory) return
    const reader = (entry as FileSystemDirectoryEntry).createReader()
    // readEntries returns at most ~100 entries per call; keep asking until it comes back empty.
    for (;;) {
      const batch = await readBatch(reader)
      if (batch.length === 0) break
      await Promise.all(batch.map((child) => walk(child, sources, failed)))
    }
  } catch {
    failed.push({ path: entry.fullPath, reason: 'Could not be read' })
  }
}

/** Reads a drop, descending into dropped folders. Must be called synchronously from the `drop` handler. */
export async function sourcesFromDataTransfer(dt: DataTransfer): Promise<{ sources: SourceFile[]; failed: Skipped[] }> {
  // `DataTransferItem`s are only valid during the event: grab every entry before the first await.
  const roots: FileSystemEntry[] = []
  const sources: SourceFile[] = []
  for (const item of Array.from(dt.items)) {
    if (item.kind !== 'file') continue
    const entry = item.webkitGetAsEntry()
    if (entry) roots.push(entry)
    else {
      const file = item.getAsFile()
      if (file) sources.push({ file, path: file.name })
    }
  }
  const failed: Skipped[] = []
  await Promise.all(roots.map((root) => walk(root, sources, failed)))
  return { sources, failed }
}

/** Unpacks archives and normalizes every path. Never throws for a bad archive: it lands in `skipped`. */
export async function ingest(sources: SourceFile[], { signal, onProgress }: IngestOptions = {}): Promise<IngestResult> {
  const result: IngestResult = { entries: [], skipped: [], junk: 0 }

  for (const [index, { file, path }] of sources.entries()) {
    signal?.throwIfAborted()
    const safe = normalizePath(path)
    if (!safe.ok) {
      result.skipped.push({ path, reason: safe.reason })
      continue
    }
    const overall = index / sources.length

    const claimed = archiveFormatFromName(file.name)
    if (claimed) {
      const format = sniffArchive(new Uint8Array(await file.slice(0, 512).arrayBuffer()))
      if (!format) {
        result.skipped.push({ path, reason: `Not a valid ${claimed === 'gzip' ? 'gzip/tar.gz' : claimed} archive` })
        continue
      }
      const dirs = safe.segments.slice(0, -1)
      try {
        const extracted = await extractArchive(file, format, {
          signal,
          onProgress: (fraction) =>
            onProgress?.({ label: `Extracting ${file.name}`, fraction: (index + fraction) / sources.length }),
        })
        for (const entry of extracted.entries) result.entries.push({ path: [...dirs, ...entry.path], file: entry.file })
        for (const s of extracted.skipped) result.skipped.push({ path: `${file.name}/${s.path}`, reason: s.reason })
        result.junk += extracted.junk
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') throw error
        result.skipped.push({ path, reason: error instanceof Error ? error.message : 'Could not be read' })
      }
      continue
    }

    if (isJunkPath(safe.segments)) result.junk++
    else result.entries.push({ path: safe.segments, file })
    if (index % 500 === 0) onProgress?.({ label: 'Reading files', fraction: overall })
  }
  return result
}

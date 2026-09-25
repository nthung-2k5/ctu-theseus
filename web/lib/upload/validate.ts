/**
 * Per-file checks run before anything is staged: size, type by extension, and type by content.
 *
 * Errors keep a file out of the upload; warnings are shown but the file still goes. Content is sniffed from
 * the first bytes because files pulled out of an archive carry no MIME type of their own.
 */

import { extensionOf, MAX_FILE_BYTES, mimeFromName, sniffMime } from './paths'
import type { Issue } from './types'

const MIME_LABEL: Record<string, string> = {
  'image/jpeg': 'JPEG',
  'image/png': 'PNG',
  'audio/wav': 'WAV',
  'audio/mpeg': 'MP3',
  'audio/flac': 'FLAC',
  'audio/ogg': 'OGG',
}

const label = (mime: string) => MIME_LABEL[mime] ?? mime

function formatMegabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** Checks one file. `accept` is the task's list of MIME types; `undefined` = anything goes. */
export async function validateFile(file: File, accept: string[] | undefined): Promise<Issue[]> {
  const issues: Issue[] = []
  if (file.size === 0) return [{ severity: 'error', message: 'Empty file' }]
  if (file.size > MAX_FILE_BYTES) {
    issues.push({
      severity: 'error',
      message: `Larger than the ${MAX_FILE_BYTES / 1024 / 1024} MB limit (${formatMegabytes(file.size)})`,
    })
  }
  if (!accept) return issues

  const byName = mimeFromName(file.name) || file.type
  if (!byName || !accept.includes(byName)) {
    const ext = extensionOf(file.name)
    issues.push({
      severity: 'error',
      message: `${ext ? `.${ext} files are` : 'Files without an extension are'} not supported here (expected ${accept.map(label).join(', ')})`,
    })
    return issues
  }

  const sniffed = sniffMime(new Uint8Array(await file.slice(0, 12).arrayBuffer()))
  if (sniffed === null) {
    issues.push({ severity: 'warning', message: `Content doesn't look like a valid ${label(byName)} file` })
  } else if (sniffed !== byName) {
    issues.push(
      accept.includes(sniffed)
        ? { severity: 'warning', message: `Named like ${label(byName)} but the content is ${label(sniffed)}` }
        : { severity: 'error', message: `Content is ${label(sniffed)}, which this task doesn't accept` },
    )
  }
  return issues
}

/** Validates many files with bounded concurrency; results line up with `files`. */
export async function validateFiles(
  files: File[],
  accept: string[] | undefined,
  options: { signal?: AbortSignal; onProgress?: (fraction: number) => void } = {},
): Promise<Issue[][]> {
  const { signal, onProgress } = options
  const out: Issue[][] = new Array(files.length)
  const STEP = 200
  for (let start = 0; start < files.length; start += STEP) {
    signal?.throwIfAborted()
    const slice = files.slice(start, start + STEP)
    const results = await Promise.all(slice.map((file) => validateFile(file, accept)))
    results.forEach((issues, i) => {
      out[start + i] = issues
    })
    onProgress?.(Math.min(1, (start + STEP) / files.length))
  }
  return out
}

async function sha256(file: File): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

/**
 * Finds files with identical content. Only files that share a size are hashed, so a batch of distinct
 * images costs nothing. Returns `id -> id of the first copy` for every later copy.
 */
export async function findDuplicates(
  files: { id: string; file: File }[],
  signal?: AbortSignal,
): Promise<Map<string, string>> {
  const duplicates = new Map<string, string>()
  if (!globalThis.crypto?.subtle) return duplicates

  const bySize = new Map<number, { id: string; file: File }[]>()
  for (const entry of files) {
    if (entry.file.size === 0) continue
    const group = bySize.get(entry.file.size)
    if (group) group.push(entry)
    else bySize.set(entry.file.size, [entry])
  }

  for (const group of bySize.values()) {
    if (group.length < 2) continue
    const firstByHash = new Map<string, string>()
    for (const { id, file } of group) {
      signal?.throwIfAborted()
      const hash = await sha256(file)
      const first = firstByHash.get(hash)
      if (first) duplicates.set(id, first)
      else firstByHash.set(hash, id)
    }
  }
  return duplicates
}

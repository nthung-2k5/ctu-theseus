/**
 * Per-file checks run before anything is staged: size, type by extension, and type by content.
 *
 * Errors keep a file out of the upload; warnings are shown but the file still goes. Content is sniffed from
 * the first bytes because files pulled out of an archive carry no MIME type of their own.
 */

import { extensionOf, maxBytesFor, megabytes, mimeFromName, sniffMime } from './paths'
import type { Issue } from './types'

const MIME_LABEL: Record<string, string> = {
  'image/jpeg': 'JPEG',
  'image/png': 'PNG',
  'audio/wav': 'WAV',
  'audio/mpeg': 'MP3',
  'audio/flac': 'FLAC',
  'audio/ogg': 'OGG',
  'video/mp4': 'MP4',
  'video/webm': 'WebM',
}

const label = (mime: string) => MIME_LABEL[mime] ?? mime

function formatMegabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** Checks one file. `accept` is the task's list of MIME types; `undefined` = anything goes. */
export async function validateFile(file: File, accept: string[] | undefined): Promise<Issue[]> {
  const issues: Issue[] = []
  if (file.size === 0) return [{ severity: 'error', message: 'Empty file' }]
  const limit = maxBytesFor(file.name)
  if (file.size > limit) {
    issues.push({
      severity: 'error',
      message: `Larger than the ${megabytes(limit)} MB limit (${formatMegabytes(file.size)})`,
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

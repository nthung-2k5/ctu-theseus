/**
 * Path and file-type helpers shared by the ingest pipeline, the archive worker and the tests.
 *
 * Pure and dependency-free on purpose: the worker bundles this file, and `bun test` runs it without the
 * `@public` alias.
 */

/** The server rejects anything larger (`MAX_UPLOAD_BYTES` in `routers/datasets.py`). */
export const MAX_FILE_BYTES = 50 * 1024 * 1024

const EXT_MIME: Record<string, string> = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  wav: 'audio/wav',
  mp3: 'audio/mpeg',
  flac: 'audio/flac',
  ogg: 'audio/ogg',
}

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : ''
}

/** The MIME type implied by a file name, or `''` when we don't know it. */
export function mimeFromName(name: string): string {
  return EXT_MIME[extensionOf(name)] ?? ''
}

export type SafePath = { ok: true; segments: string[] } | { ok: false; reason: string }

/**
 * Splits an archive/drop path into clean segments. Empty and `.` segments vanish; a `..` segment rejects
 * the whole entry, so nothing can claim to live outside the folder tree it was dropped into.
 */
export function normalizePath(raw: string): SafePath {
  const segments: string[] = []
  for (const part of raw.split(/[\\/]+/)) {
    if (part === '' || part === '.') continue
    if (part === '..') return { ok: false, reason: 'Path escapes the folder (contains "..")' }
    segments.push(part)
  }
  if (segments.length === 0) return { ok: false, reason: 'Empty path' }
  return { ok: true, segments }
}

const JUNK_NAMES = new Set(['thumbs.db', 'desktop.ini', '__macosx'])

/** OS metadata that shows up in archives and folder drops and is never dataset content. */
export function isJunkPath(segments: string[]): boolean {
  return segments.some((s) => s.startsWith('.') || JUNK_NAMES.has(s.toLowerCase()))
}

/** Normalizes a class/folder name for matching: `Golden_Retriever`, `golden-retriever` and `golden retriever` agree. */
export function normalizeClassName(name: string): string {
  return name
    .trim()
    .replace(/[_\-\s]+/g, ' ')
    .toLowerCase()
}

export type ArchiveFormat = 'zip' | 'gzip' | 'tar'

/** What an archive's *name* claims it is. Confirmed against magic bytes before anything is extracted. */
export function archiveFormatFromName(name: string): ArchiveFormat | null {
  const lower = name.toLowerCase()
  if (lower.endsWith('.zip')) return 'zip'
  if (lower.endsWith('.tar.gz') || lower.endsWith('.tgz') || lower.endsWith('.gz')) return 'gzip'
  if (lower.endsWith('.tar')) return 'tar'
  return null
}

/** Confirms an archive's real format from its first bytes. `null` = not an archive we can read. */
export function sniffArchive(head: Uint8Array): ArchiveFormat | null {
  if (head.length >= 4 && head[0] === 0x50 && head[1] === 0x4b && (head[2] === 3 || head[2] === 5)) return 'zip'
  if (head.length >= 2 && head[0] === 0x1f && head[1] === 0x8b) return 'gzip'
  if (looksLikeTar(head)) return 'tar'
  return null
}

export function looksLikeTar(head: Uint8Array): boolean {
  if (head.length < 263) return false
  const magic = String.fromCharCode(...head.subarray(257, 262))
  return magic === 'ustar'
}

/** Sniffs an accepted media type from a file's first bytes; `null` when it matches none of them. */
export function sniffMime(head: Uint8Array): string | null {
  const b = head
  if (b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'image/jpeg'
  if (b.length >= 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return 'image/png'
  const ascii = (from: number, to: number) => String.fromCharCode(...b.subarray(from, to))
  if (b.length >= 12 && ascii(0, 4) === 'RIFF' && ascii(8, 12) === 'WAVE') return 'audio/wav'
  if (b.length >= 4 && ascii(0, 4) === 'fLaC') return 'audio/flac'
  if (b.length >= 4 && ascii(0, 4) === 'OggS') return 'audio/ogg'
  if (b.length >= 3 && ascii(0, 3) === 'ID3') return 'audio/mpeg'
  if (b.length >= 2 && b[0] === 0xff && (b[1] & 0xe0) === 0xe0) return 'audio/mpeg'
  return null
}

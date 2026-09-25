/**
 * A small streaming reader for ustar/GNU/pax tar archives.
 *
 * Push decompressed bytes in any chunking; the handlers fire per entry. It never holds more than one
 * 512-byte header plus the (tiny) body of a long-name/pax record, so a multi-gigabyte `.tar.gz` streams
 * through without being buffered. Entry *bodies* are handed to `onData` in whatever chunks arrive.
 */

export interface TarHandlers {
  /** A regular file starts. Return `false` to skip its body (too big, junk, unsafe path...). */
  onFile(path: string, size: number): boolean
  /** A slice of the current file's body. Only called after `onFile` returned `true`. */
  onData(chunk: Uint8Array): void
  /** The current file's body is complete. */
  onFileEnd(): void
  /** An entry we can't represent (symlink, device...) was skipped. */
  onSkipped(path: string, reason: string): void
}

const BLOCK = 512
const MAX_META_BYTES = 1024 * 1024

const decoder = new TextDecoder()

function cString(bytes: Uint8Array): string {
  const end = bytes.indexOf(0)
  return decoder.decode(end === -1 ? bytes : bytes.subarray(0, end))
}

function parseNumber(bytes: Uint8Array): number {
  // GNU base-256 encoding for values that overflow the octal field.
  if (bytes[0] & 0x80) {
    let value = bytes[0] & 0x7f
    for (let i = 1; i < bytes.length; i++) value = value * 256 + bytes[i]
    return value
  }
  const text = cString(bytes).trim()
  if (text === '') return 0
  const value = Number.parseInt(text, 8)
  if (Number.isNaN(value)) throw new Error('Not a valid tar archive')
  return value
}

/** pax records look like `<len> <key>=<value>\n`; only `path` and `size` matter to us. */
function parsePax(data: Uint8Array): { path?: string; size?: number } {
  const out: { path?: string; size?: number } = {}
  let pos = 0
  while (pos < data.length) {
    const space = data.indexOf(0x20, pos)
    if (space === -1) break
    const len = Number.parseInt(decoder.decode(data.subarray(pos, space)), 10)
    if (!Number.isFinite(len) || len <= 0) break
    const record = decoder.decode(data.subarray(space + 1, pos + len - 1))
    const eq = record.indexOf('=')
    if (eq !== -1) {
      const key = record.slice(0, eq)
      const value = record.slice(eq + 1)
      if (key === 'path') out.path = value
      else if (key === 'size') out.size = Number.parseInt(value, 10)
    }
    pos += len
  }
  return out
}

type Mode = 'header' | 'body' | 'meta' | 'skip'

export class TarParser {
  private readonly header = new Uint8Array(BLOCK)
  private headerLen = 0
  private mode: Mode = 'header'
  private remaining = 0
  private padding = 0
  private ended = false
  private sawHeader = false

  private metaKind: 'L' | 'x' | null = null
  private metaChunks: Uint8Array[] = []
  private longName: string | null = null
  private pax: { path?: string; size?: number } | null = null
  private readonly handlers: TarHandlers

  constructor(handlers: TarHandlers) {
    this.handlers = handlers
  }

  push(chunk: Uint8Array): void {
    let pos = 0
    while (pos < chunk.length && !this.ended) {
      if (this.mode === 'header') {
        const take = Math.min(BLOCK - this.headerLen, chunk.length - pos)
        this.header.set(chunk.subarray(pos, pos + take), this.headerLen)
        this.headerLen += take
        pos += take
        if (this.headerLen === BLOCK) {
          this.headerLen = 0
          this.onHeader()
        }
        continue
      }
      const take = Math.min(this.remaining, chunk.length - pos)
      const slice = chunk.subarray(pos, pos + take)
      if (this.mode === 'body') this.handlers.onData(slice)
      else if (this.mode === 'meta') this.metaChunks.push(slice.slice())
      this.remaining -= take
      pos += take
      if (this.remaining === 0) this.finishRegion()
    }
  }

  /** Call when the stream ends; throws if it stopped mid-entry. */
  end(): void {
    if (this.ended) return
    const midEntry = this.mode !== 'header' && this.remaining > 0
    if (this.headerLen > 0 || midEntry) throw new Error('Archive is truncated')
  }

  /** True once at least one valid header (or the end marker) was read. */
  get isTar(): boolean {
    return this.sawHeader || this.ended
  }

  private onHeader(): void {
    const h = this.header
    if (h.every((b) => b === 0)) {
      this.ended = true
      return
    }

    let sum = 0
    for (let i = 0; i < BLOCK; i++) sum += i >= 148 && i < 156 ? 0x20 : h[i]
    if (sum !== parseNumber(h.subarray(148, 156))) throw new Error('Not a valid tar archive')
    this.sawHeader = true

    const typeflag = h[156] === 0 ? '0' : String.fromCharCode(h[156])
    let size = parseNumber(h.subarray(124, 136))
    const isUstar = cString(h.subarray(257, 262)) === 'ustar'
    const prefix = isUstar ? cString(h.subarray(345, 500)) : ''
    const name = cString(h.subarray(0, 100))
    let path = prefix ? `${prefix}/${name}` : name

    if (typeflag === 'L' || typeflag === 'x') {
      if (size > MAX_META_BYTES) throw new Error('Not a valid tar archive')
      this.metaKind = typeflag
      this.metaChunks = []
      this.startRegion('meta', size)
      return
    }
    if (typeflag === 'g') {
      this.startRegion('skip', size)
      return
    }

    if (this.longName !== null) path = this.longName
    if (this.pax?.path !== undefined) path = this.pax.path
    if (this.pax?.size !== undefined && Number.isFinite(this.pax.size)) size = this.pax.size
    this.longName = null
    this.pax = null

    if (typeflag === '0' || typeflag === '7') {
      if (this.handlers.onFile(path, size)) {
        this.startRegion('body', size)
        return
      }
    } else if (typeflag !== '5') {
      const linkLike = typeflag === '1' || typeflag === '2'
      this.handlers.onSkipped(path, linkLike ? 'Links are not supported' : 'Unsupported entry type')
    }
    this.startRegion('skip', typeflag === '5' ? 0 : size)
  }

  private startRegion(mode: Mode, size: number): void {
    const padding = (BLOCK - (size % BLOCK)) % BLOCK
    this.mode = mode
    // A skipped region swallows its own padding; a read one hands it to `finishRegion` afterwards.
    this.remaining = mode === 'skip' ? size + padding : size
    this.padding = mode === 'skip' ? 0 : padding
    if (this.remaining === 0) this.finishRegion()
  }

  private finishRegion(): void {
    if (this.mode === 'body') this.handlers.onFileEnd()
    else if (this.mode === 'meta') this.applyMeta()

    if (this.mode !== 'skip' && this.padding > 0) {
      // Consume the padding up to the next 512-byte boundary; the skip that follows has no padding of its own.
      this.mode = 'skip'
      this.remaining = this.padding
      this.padding = 0
      return
    }
    this.mode = 'header'
    this.padding = 0
  }

  private applyMeta(): void {
    let total = 0
    for (const c of this.metaChunks) total += c.length
    const data = new Uint8Array(total)
    let offset = 0
    for (const c of this.metaChunks) {
      data.set(c, offset)
      offset += c.length
    }
    this.metaChunks = []
    if (this.metaKind === 'L') this.longName = cString(data)
    else this.pax = parsePax(data)
    this.metaKind = null
  }
}

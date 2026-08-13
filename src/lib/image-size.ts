/**
 * Minimal PNG/JPEG dimension reader.
 */

export interface ImageDimensions {
  width: number
  height: number
}

export function readImageDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (isPng(bytes)) return readPngDimensions(bytes)
  if (isJpeg(bytes)) return readJpegDimensions(bytes)
  return null
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

function isPng(bytes: Uint8Array): boolean {
  return PNG_SIGNATURE.every((byte, i) => bytes[i] === byte)
}

function readPngDimensions(bytes: Uint8Array): ImageDimensions | null {
  if (bytes.length < 24) return null
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  // IHDR is always the first chunk: 8-byte signature, 4-byte length, 4-byte "IHDR", then width/height (big-endian u32).
  return { width: view.getUint32(16, false), height: view.getUint32(20, false) }
}

function isJpeg(bytes: Uint8Array): boolean {
  return bytes[0] === 0xff && bytes[1] === 0xd8
}

function readJpegDimensions(bytes: Uint8Array): ImageDimensions | null {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let offset = 2
  while (offset + 9 <= bytes.length) {
    if (view.getUint8(offset) !== 0xff) return null
    const marker = view.getUint8(offset + 1)
    // SOFn markers (start of frame) carry the dimensions; SOF4/8/12 are DHT/JPG/DAC, not frames.
    const isSofMarker = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc
    if (isSofMarker) {
      return { height: view.getUint16(offset + 5, false), width: view.getUint16(offset + 7, false) }
    }
    const segmentLength = view.getUint16(offset + 2, false)
    offset += 2 + segmentLength
  }
  return null
}

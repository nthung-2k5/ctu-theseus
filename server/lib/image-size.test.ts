import { describe, expect, test } from 'bun:test'
import { readImageDimensions } from './image-size'

describe('readImageDimensions', () => {
  test('reads width/height from a PNG IHDR chunk', () => {
    // biome-ignore format: byte-per-line layout matches the PNG structure it documents
    const png = new Uint8Array([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // signature
      0x00, 0x00, 0x00, 0x0d, // IHDR chunk length (13)
      0x49, 0x48, 0x44, 0x52, // "IHDR"
      0x00, 0x00, 0x00, 0x64, // width = 100
      0x00, 0x00, 0x00, 0x32, // height = 50
    ])

    expect(readImageDimensions(png)).toEqual({ width: 100, height: 50 })
  })

  test('reads width/height from a JPEG SOF0 marker', () => {
    // biome-ignore format: byte-per-line layout matches the JPEG structure it documents
    const jpeg = new Uint8Array([
      0xff, 0xd8, // SOI
      0xff, 0xc0, // SOF0
      0x00, 0x0b, // segment length
      0x08, // precision
      0x00, 0x32, // height = 50
      0x00, 0x64, // width = 100
    ])

    expect(readImageDimensions(jpeg)).toEqual({ width: 100, height: 50 })
  })

  test('skips non-SOF JPEG segments (e.g. APP0) to find SOF0', () => {
    // biome-ignore format: byte-per-line layout matches the JPEG structure it documents
    const jpeg = new Uint8Array([
      0xff, 0xd8, // SOI
      0xff, 0xe0, 0x00, 0x04, 0xab, 0xcd, // APP0, length 4 (2 content bytes)
      0xff, 0xc0, // SOF0
      0x00, 0x0b,
      0x08,
      0x00, 0x0a, // height = 10
      0x00, 0x14, // width = 20
    ])

    expect(readImageDimensions(jpeg)).toEqual({ width: 20, height: 10 })
  })

  test('returns null for an unrecognized format', () => {
    expect(readImageDimensions(new Uint8Array([0x00, 0x01, 0x02, 0x03]))).toBeNull()
  })

  test('returns null for a truncated PNG', () => {
    const truncated = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    expect(readImageDimensions(truncated)).toBeNull()
  })
})

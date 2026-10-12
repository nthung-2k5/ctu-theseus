import { expect, test } from 'bun:test'
import { findDuplicates, validateFile } from './validate'

const IMAGES = ['image/jpeg', 'image/png']
const VIDEOS = ['video/mp4', 'video/webm']
const PNG = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0])
const JPEG = Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0])
// A 4-byte box size, `ftyp`, then a brand.
const MP4 = Uint8Array.from([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d])
const WEBM = Uint8Array.from([0x1a, 0x45, 0xdf, 0xa3, 0, 0, 0, 0, 0, 0, 0, 0])

test('accepts a real png', async () => {
  expect(await validateFile(new File([PNG], 'a.png'), IMAGES)).toEqual([])
})

test('rejects unsupported extensions, empty and oversized files', async () => {
  const [txt] = await validateFile(new File(['hello'], 'notes.txt'), IMAGES)
  expect(txt.severity).toBe('error')
  expect(txt.message).toContain('.txt')
  expect((await validateFile(new File([], 'a.png'), IMAGES))[0].message).toBe('Empty file')
  const big = new File([new Uint8Array(50 * 1024 * 1024 + 1)], 'big.png')
  expect((await validateFile(big, IMAGES))[0].message).toContain('50 MB')
})

test('warns when content and name disagree, errors when the content is unsupported', async () => {
  expect((await validateFile(new File([JPEG], 'a.png'), IMAGES))[0].severity).toBe('warning')
  expect((await validateFile(new File(['not really an image'], 'a.png'), IMAGES))[0].severity).toBe('warning')
  const wav = new File([new TextEncoder().encode('RIFF0000WAVEfmt ')], 'a.png')
  expect((await validateFile(wav, IMAGES))[0].severity).toBe('error')
})

test('any file passes when the task has no accept list', async () => {
  expect(await validateFile(new File(['x'], 'whatever.bin'), undefined)).toEqual([])
})

test('finds duplicates by content, not just size', async () => {
  const files = [
    { id: 'a', file: new File(['same!'], 'a.png') },
    { id: 'b', file: new File(['same!'], 'b.png') },
    { id: 'c', file: new File(['other'], 'c.png') },
    { id: 'd', file: new File(['unique-size'], 'd.png') },
  ]
  expect([...(await findDuplicates(files))]).toEqual([['b', 'a']])
})

test('accepts real mp4 and webm clips for a video task', async () => {
  expect(await validateFile(new File([MP4], 'walk.mp4'), VIDEOS)).toEqual([])
  expect(await validateFile(new File([WEBM], 'walk.webm'), VIDEOS)).toEqual([])
})

test('a video task refuses images, and an image task refuses video', async () => {
  expect((await validateFile(new File([PNG], 'a.png'), VIDEOS))[0].message).toContain('expected MP4, WebM')
  expect((await validateFile(new File([MP4], 'a.mp4'), IMAGES))[0].severity).toBe('error')
})

test('video has a larger size limit than images', async () => {
  const clip = new File([MP4, new Uint8Array(60 * 1024 * 1024)], 'big.mp4')
  expect(await validateFile(clip, VIDEOS)).toEqual([]) // over the 50 MB image limit, under the 200 MB video one
  const huge = new File([MP4, new Uint8Array(200 * 1024 * 1024)], 'huge.mp4')
  expect((await validateFile(huge, VIDEOS))[0].message).toContain('200 MB')
})

test('a video whose content is not a video warns', async () => {
  expect((await validateFile(new File(['not really a video'], 'a.mp4'), VIDEOS))[0].severity).toBe('warning')
})

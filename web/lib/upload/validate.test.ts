import { expect, test } from 'bun:test'
import { findDuplicates, validateFile } from './validate'

const IMAGES = ['image/jpeg', 'image/png']
const PNG = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0])
const JPEG = Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0])

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

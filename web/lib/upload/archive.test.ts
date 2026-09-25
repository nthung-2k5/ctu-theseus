import { expect, test } from 'bun:test'
import { gzipSync, zipSync } from 'fflate'
import { extractArchive } from './archive'

const enc = new TextEncoder()

/** Minimal ustar writer, enough to build fixtures. */
function tar(files: Record<string, string | Uint8Array>): Uint8Array<ArrayBuffer> {
  const blocks: Uint8Array[] = []
  for (const [name, content] of Object.entries(files)) {
    const data = typeof content === 'string' ? enc.encode(content) : content
    const h = new Uint8Array(512)
    h.set(enc.encode(name), 0)
    h.set(enc.encode(`${data.length.toString(8).padStart(11, '0')}\0`), 124)
    h[156] = '0'.charCodeAt(0)
    h.set(enc.encode('ustar\0'), 257)
    h.set(enc.encode('        '), 148)
    let sum = 0
    for (const b of h) sum += b
    h.set(enc.encode(`${sum.toString(8).padStart(6, '0')}\0 `), 148)
    blocks.push(h, data, new Uint8Array((512 - (data.length % 512)) % 512))
  }
  blocks.push(new Uint8Array(1024))
  const out = new Uint8Array(blocks.reduce((n, b) => n + b.length, 0))
  let offset = 0
  for (const b of blocks) {
    out.set(b, offset)
    offset += b.length
  }
  return out
}

const png = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3])

const tree = {
  'pets/cat/train/a.png': png,
  'pets/cat/val/b.png': png,
  'pets/dog/test/c.png': png,
  'pets/__MACOSX/cat/._a.png': enc.encode('junk'),
  'pets/.DS_Store': enc.encode('junk'),
}

async function summarize(file: File, format: 'zip' | 'gzip' | 'tar') {
  const result = await extractArchive(file, format)
  return {
    paths: result.entries.map((e) => e.path.join('/')).sort(),
    types: new Set(result.entries.map((e) => e.file.type)),
    junk: result.junk,
    skipped: result.skipped,
    bytes: result.entries.map((e) => e.file.size),
  }
}

const expectedPaths = ['pets/cat/train/a.png', 'pets/cat/val/b.png', 'pets/dog/test/c.png']

test('extracts a zip, dropping junk and typing files by extension', async () => {
  const zip = zipSync(tree)
  const result = await summarize(new File([zip], 'pets.zip'), 'zip')
  expect(result.paths).toEqual(expectedPaths)
  expect(result.junk).toBe(2)
  expect([...result.types]).toEqual(['image/png'])
  expect(result.bytes).toEqual([png.length, png.length, png.length])
})

test('extracts a tar and a tar.gz', async () => {
  const raw = tar(tree)
  expect((await summarize(new File([raw], 'pets.tar'), 'tar')).paths).toEqual(expectedPaths)
  const gz = await summarize(new File([gzipSync(raw)], 'pets.tar.gz'), 'gzip')
  expect(gz.paths).toEqual(expectedPaths)
  expect(gz.junk).toBe(2)
})

test('a lone gzipped file becomes one entry without the .gz', async () => {
  const result = await summarize(new File([gzipSync(png)], 'photo.png.gz'), 'gzip')
  expect(result.paths).toEqual(['photo.png'])
  expect(result.bytes).toEqual([png.length])
})

test('unsafe paths and oversized members are skipped, not extracted', async () => {
  const big = new Uint8Array(51 * 1024 * 1024)
  const result = await summarize(
    new File([zipSync({ 'ok.png': png, '../evil.png': png, 'big.png': big })], 'x.zip'),
    'zip',
  )
  expect(result.paths).toEqual(['ok.png'])
  expect(result.skipped.map((s) => s.path).sort()).toEqual(['../evil.png', 'big.png'])
})

test('a corrupt tar rejects the whole archive', async () => {
  await expect(extractArchive(new File([enc.encode('x'.repeat(2048))], 'bad.tar'), 'tar')).rejects.toThrow()
})

test('aborting stops extraction', async () => {
  const controller = new AbortController()
  controller.abort()
  await expect(
    extractArchive(new File([zipSync(tree)], 'x.zip'), 'zip', { signal: controller.signal }),
  ).rejects.toThrow()
})

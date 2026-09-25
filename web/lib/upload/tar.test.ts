import { expect, test } from 'bun:test'
import { TarParser } from './tar'

const enc = new TextEncoder()

function header(name: string, size: number, typeflag = '0', prefix = ''): Uint8Array {
  const h = new Uint8Array(512)
  h.set(enc.encode(name).subarray(0, 100), 0)
  h.set(enc.encode(`${size.toString(8).padStart(11, '0')}\0`), 124)
  h[156] = typeflag.charCodeAt(0)
  h.set(enc.encode('ustar\0'), 257)
  h.set(enc.encode(prefix), 345)
  h.set(enc.encode('        '), 148)
  let sum = 0
  for (const b of h) sum += b
  h.set(enc.encode(`${sum.toString(8).padStart(6, '0')}\0 `), 148)
  return h
}

function pad(data: Uint8Array): Uint8Array {
  const out = new Uint8Array(Math.ceil(data.length / 512) * 512)
  out.set(data)
  return out
}

function concat(...parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out
}

function file(name: string, body: string, prefix = ''): Uint8Array {
  const data = enc.encode(body)
  return concat(header(name, data.length, '0', prefix), pad(data))
}

function parse(bytes: Uint8Array, chunkSize = bytes.length, keep: (path: string) => boolean = () => true) {
  const files: { path: string; body: string }[] = []
  const skipped: string[] = []
  let current: { path: string; parts: Uint8Array[] } | null = null
  const parser = new TarParser({
    onFile: (path) => {
      if (!keep(path)) return false
      current = { path, parts: [] }
      return true
    },
    onData: (chunk) => current?.parts.push(chunk.slice()),
    onFileEnd: () => {
      if (current) files.push({ path: current.path, body: new TextDecoder().decode(concat(...current.parts)) })
      current = null
    },
    onSkipped: (path) => skipped.push(path),
  })
  for (let i = 0; i < bytes.length; i += chunkSize) parser.push(bytes.subarray(i, i + chunkSize))
  parser.end()
  return { files, skipped }
}

const END = new Uint8Array(1024)

test('reads files, whatever the chunking', () => {
  const tar = concat(file('cat/train/a.jpg', 'hello'), file('cat/test/b.jpg', 'x'.repeat(700)), END)
  for (const chunk of [tar.length, 512, 100, 7]) {
    const { files } = parse(tar, chunk)
    expect(files.map((f) => f.path)).toEqual(['cat/train/a.jpg', 'cat/test/b.jpg'])
    expect(files[0].body).toBe('hello')
    expect(files[1].body).toBe('x'.repeat(700))
  }
})

test('joins ustar prefix and name', () => {
  const { files } = parse(concat(file('a.jpg', 'x', 'dogs/train'), END))
  expect(files[0].path).toBe('dogs/train/a.jpg')
})

test('uses pax and GNU long names', () => {
  const long = `${'d/'.repeat(80)}file.png`
  const record = ` path=${long}\n`
  const paxBody = enc.encode(`${record.length + String(record.length).length} path=${long}\n`)
  const pax = concat(header('PaxHeader', paxBody.length, 'x'), pad(paxBody))
  const gnuBody = enc.encode(`${long}\0`)
  const gnu = concat(header('././@LongLink', gnuBody.length, 'L'), pad(gnuBody))
  expect(parse(concat(pax, file('short', 'x'), END)).files[0].path).toBe(long)
  expect(parse(concat(gnu, file('short', 'x'), END)).files[0].path).toBe(long)
})

test('skips bodies the handler declines and reports links', () => {
  const link = header('cat/link.jpg', 0, '2')
  const tar = concat(file('big.jpg', 'x'.repeat(2000)), link, file('ok.jpg', 'ok'), END)
  const { files, skipped } = parse(tar, 300, (path) => path !== 'big.jpg')
  expect(files.map((f) => f.path)).toEqual(['ok.jpg'])
  expect(skipped).toEqual(['cat/link.jpg'])
})

test('rejects non-tar data and truncated archives', () => {
  expect(() => parse(enc.encode('x'.repeat(1024)))).toThrow('Not a valid tar archive')
  const tar = file('a.jpg', 'x'.repeat(2000))
  expect(() => parse(tar.subarray(0, 700))).toThrow('truncated')
})

import { expect, test } from 'bun:test'
import { strToU8, zipSync } from 'fflate'
import { readZipConfig } from './modelZip'

const zip = (files: Record<string, string | Uint8Array>) =>
  new Blob([
    zipSync(Object.fromEntries(Object.entries(files).map(([k, v]) => [k, typeof v === 'string' ? strToU8(v) : v]))),
  ])

const CONFIG = '{"model_type": "bert", "architectures": ["BertModel"]}'

test('reads config.json at the top of the zip', async () => {
  expect(await readZipConfig(zip({ 'config.json': CONFIG, 'model.safetensors': new Uint8Array(64) }))).toEqual({
    model_type: 'bert',
    architectures: ['BertModel'],
  })
})

test('reads config.json inside the single folder that holds the model', async () => {
  expect(
    await readZipConfig(zip({ 'my-model/config.json': CONFIG, 'my-model/w.safetensors': new Uint8Array(8) })),
  ).toEqual({
    model_type: 'bert',
    architectures: ['BertModel'],
  })
})

test('finds config.json after a large weights file', async () => {
  const blob = zip({ 'a.safetensors': new Uint8Array(1_000_000), 'config.json': CONFIG })
  expect((await readZipConfig(blob))?.model_type).toBe('bert')
})

test('is null when there is no config.json, it is deeper than one folder, or it is not an object', async () => {
  expect(await readZipConfig(zip({ 'w.safetensors': new Uint8Array(8) }))).toBeNull()
  expect(await readZipConfig(zip({ 'a/b/config.json': CONFIG }))).toBeNull()
  expect(await readZipConfig(zip({ 'config.json': '[1, 2]' }))).toBeNull()
  expect(await readZipConfig(zip({ 'config.json': 'not json' }))).toBeNull()
})

test('is null for a file that is not a zip', async () => {
  expect(await readZipConfig(new Blob(['definitely not a zip']))).toBeNull()
})

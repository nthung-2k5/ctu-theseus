import { Unzip, UnzipInflate, UnzipPassThrough } from 'fflate'

/** How much of the zip is read looking for config.json before giving up (it normally sits at the start). */
const MAX_SCAN_BYTES = 256 * 1024 * 1024
const MAX_CONFIG_BYTES = 2 * 1024 * 1024

/** `config.json` at the top of the zip, or inside the single folder that holds the model (`my-model/config.json`). */
const isConfigPath = (name: string) => /^([^/]+\/)?config\.json$/.test(name) && !name.startsWith('__MACOSX/')

/**
 * Read the model's `config.json` out of a zip without uploading it, so the form can say what the model is
 * before anything is sent. The zip is streamed entry by entry and reading stops as soon as the config is
 * found, so a multi-gigabyte zip costs only what precedes `config.json`.
 *
 * Returns null when there is no readable config.json in the part of the zip that was scanned (or the file is
 * not a zip at all): that is not an error here, the server checks the real thing after the upload.
 */
export async function readZipConfig(file: Blob): Promise<Record<string, unknown> | null> {
  let result: Record<string, unknown> | null = null
  let done = false

  const unzip = new Unzip()
  unzip.register(UnzipInflate)
  unzip.register(UnzipPassThrough)
  unzip.onfile = (entry) => {
    if (done || !isConfigPath(entry.name)) return // never started, so fflate skips the entry's bytes
    if (entry.originalSize !== undefined && entry.originalSize > MAX_CONFIG_BYTES) return
    const chunks: Uint8Array[] = []
    let size = 0
    entry.ondata = (error, chunk, final) => {
      if (done) return
      if (error) {
        done = true
        return
      }
      size += chunk.length
      if (size > MAX_CONFIG_BYTES) {
        done = true
        entry.terminate()
        return
      }
      chunks.push(chunk)
      if (!final) return
      done = true
      try {
        const parsed: unknown = JSON.parse(new TextDecoder().decode(concat(chunks, size)))
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) result = parsed as Record<string, unknown>
      } catch {
        // not JSON: reported as "no config.json" like a missing one
      }
    }
    entry.start()
  }

  try {
    const reader = file.stream().getReader()
    let scanned = 0
    while (!done && scanned < MAX_SCAN_BYTES) {
      const { value, done: finished } = await reader.read()
      if (finished) {
        unzip.push(new Uint8Array(0), true)
        break
      }
      scanned += value.length
      unzip.push(value, false)
    }
    await reader.cancel()
  } catch {
    // a corrupt or non-zip file: nothing readable
  }
  return result
}

function concat(chunks: Uint8Array[], size: number): Uint8Array {
  const out = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

/**
 * Assembles an export bundle's zip and uploads it. Called by the queue
 * (lib/export/queue.ts) once the underlying model artifact exists — either
 * because it already did (model tier re-export) or because the worker's
 * conversion finished (see the `export` case in lib/microservice.ts).
 */

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { modelExports } from '@server/db/schema'
import type { ExportFormat } from '@server/lib/enums'
import { bundleKey, downloadFile, exportKey, fileExists, uploadFile } from '@server/lib/storage'
import { getTaskDescriptor } from '@server/lib/tasks'
import { record } from '@server/lib/telemetry'
import { eq } from 'drizzle-orm'
import { strToU8, zipSync } from 'fflate'
import { extractPreprocessing } from './metadata'
import { renderReadme } from './readme'
import { render, templates } from './templates'

interface ExpectedJson {
  schemaVersion: 1
  inputColumn: string
  inputValue: unknown
  outputColumn: string
  outputType: string
  predictions: Record<string, number>
}

interface ResolvedSample {
  /** e.g. "input.jpg", "input.txt", "input.json" */
  filename: string
  bytes: Uint8Array
}

/**
 * Turn expected.json's raw `inputValue` (an `s3://` URI for file-backed
 * modalities, or an inline text/scalar value otherwise) into an actual file
 * to embed in the bundle under `sample/`.
 */
async function resolveSampleFile(inputValue: unknown): Promise<ResolvedSample | null> {
  if (typeof inputValue === 'string' && inputValue.startsWith('s3://')) {
    const withoutScheme = inputValue.slice('s3://'.length)
    const slashIdx = withoutScheme.indexOf('/')
    if (slashIdx === -1) return null
    const bucket = withoutScheme.slice(0, slashIdx)
    const key = withoutScheme.slice(slashIdx + 1)
    const dotIdx = key.lastIndexOf('.')
    const ext = dotIdx === -1 ? '' : key.slice(dotIdx)
    try {
      const bytes = await downloadFile(bucket, key)
      return { filename: `input${ext}`, bytes }
    } catch {
      return null
    }
  }
  if (typeof inputValue === 'string') {
    return { filename: 'input.txt', bytes: strToU8(inputValue) }
  }
  if (inputValue !== undefined && inputValue !== null) {
    return { filename: 'input.json', bytes: strToU8(JSON.stringify(inputValue, null, 2)) }
  }
  return null
}

/** Load the worker-written expected.json (if any) and its resolved sample file, rewritten to bundle-local paths. */
async function loadGoldenSample(runId: string): Promise<{ expectedJson: Uint8Array; sample: ResolvedSample } | null> {
  const exists = await fileExists(CONSTANTS.BUCKET_MODELS, `${runId}/expected.json`)
  if (!exists) return null

  const raw = await downloadFile(CONSTANTS.BUCKET_MODELS, `${runId}/expected.json`)
  const expected = JSON.parse(new TextDecoder().decode(raw)) as ExpectedJson
  const sample = await resolveSampleFile(expected.inputValue)
  if (!sample) return null

  const bundleLocal = {
    schemaVersion: 1 as const,
    sampleFile: `sample/${sample.filename}`,
    outputColumn: expected.outputColumn,
    outputType: expected.outputType,
    predictions: expected.predictions,
  }
  return { expectedJson: strToU8(JSON.stringify(bundleLocal, null, 2)), sample }
}

/** A Dart package/app name must be lowercase_with_underscores and start with a letter — pubspec.yaml's `name:` rejects anything else. */
function dartPackageName(name: string): string {
  const snake = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  const safe = /^[a-z]/.test(snake) ? snake : `app_${snake}`
  return safe || 'theseus_app'
}

type ZipInput = Parameters<typeof zipSync>[0]

async function assembleFiles(exportId: string): Promise<ZipInput> {
  const row = await db.query.modelExports.findFirst({
    where: { id: exportId },
    with: { run: { with: { project: true } } },
  })
  if (!row) throw new Error(`Export ${exportId} not found`)
  const { run, tier, format, lang } = row
  if (!run.project) throw new Error(`Run ${run.id}'s project not found`)

  const task = getTaskDescriptor(run.project.task)
  const modelBytes = await downloadFile(CONSTANTS.BUCKET_MODELS, exportKey(run.id, format as ExportFormat))
  const preprocessing = await extractPreprocessing(run.id)
  const preprocessingJson = strToU8(JSON.stringify(preprocessing, null, 2))
  const labelsTxt = preprocessing.outputs.find((o) => o.classes)?.classes?.join('\n')

  const files: ZipInput = {}

  /** Places model.<format> / preprocessing.json / labels.txt under `prefix` (root for most langs, `assets/` for Flutter). */
  function placeModelFiles(prefix: string): void {
    files[`${prefix}model.${format}`] = [modelBytes, { level: 0 }]
    files[`${prefix}preprocessing.json`] = preprocessingJson
    if (labelsTxt) files[`${prefix}labels.txt`] = strToU8(labelsTxt)
  }

  if (tier === 'model') {
    placeModelFiles('')
    files['README.md'] = strToU8(
      renderReadme(
        { runName: run.name, taskLabel: task.label, format: format as ExportFormat, tier, hasVerify: false },
        null,
      ),
    )
    return files
  }

  // devkit / app: model tier's files, plus a generated client.
  const golden = await loadGoldenSample(run.id)

  /** Places expected.json / sample/<file> under `prefix`, mirroring placeModelFiles. No-op if there's no golden sample. */
  function placeGoldenSample(prefix: string): void {
    if (!golden) return
    files[`${prefix}expected.json`] = golden.expectedJson
    files[`${prefix}sample/${golden.sample.filename}`] = golden.sample.bytes
  }

  const vars = { RUN_NAME: run.name, TASK: task.label }
  const readmeVars = {
    runName: run.name,
    taskLabel: task.label,
    format: format as ExportFormat,
    tier,
    hasVerify: !!golden,
  }

  if (lang === 'python') {
    placeModelFiles('')
    placeGoldenSample('')
    files['theseus_client.py'] = strToU8(templates.python.client)
    files['example.py'] = strToU8(templates.python.example)
    if (golden) files['verify.py'] = strToU8(templates.python.verify)
    files['README.md'] = strToU8(renderReadme(readmeVars, 'python'))
  } else if (lang === 'typescript') {
    placeModelFiles('')
    placeGoldenSample('')
    files['client.ts'] = strToU8(templates.typescript.client)
    files['example.ts'] = strToU8(templates.typescript.example)
    if (golden) files['verify.ts'] = strToU8(templates.typescript.verify)
    files['README.md'] = strToU8(renderReadme(readmeVars, 'typescript'))
  } else if (lang === 'csharp') {
    placeModelFiles('')
    placeGoldenSample('')
    files['TheseusClient.cs'] = strToU8(templates.csharp.client)
    files['Program.cs'] = strToU8(templates.csharp.program)
    files['README.md'] = strToU8(renderReadme(readmeVars, 'csharp'))
  } else if (lang === 'java') {
    placeModelFiles('')
    placeGoldenSample('')
    files['TheseusClient.java'] = strToU8(templates.java.client)
    files['Main.java'] = strToU8(templates.java.main)
    files['README.md'] = strToU8(renderReadme(readmeVars, 'java'))
  } else if (lang === 'pwa') {
    // Fetched relative to index.html at runtime — root placement is correct.
    placeModelFiles('')
    placeGoldenSample('')
    files['index.html'] = strToU8(render(templates.pwa.indexHtml, vars))
    files['app.js'] = strToU8(templates.pwa.appJs)
    files['sw.js'] = strToU8(templates.pwa.serviceWorker)
    files['manifest.webmanifest'] = strToU8(render(templates.pwa.manifest, vars))
    files['icon.svg'] = strToU8(templates.pwa.icon)
    files['style.css'] = strToU8(templates.pwa.style)
    files['README.md'] = strToU8(renderReadme(readmeVars, 'pwa'))
  } else if (lang === 'flutter') {
    // Flutter only reads files declared as pubspec assets — assets/ prefix required.
    placeModelFiles('assets/')
    placeGoldenSample('assets/')
    files['pubspec.yaml'] = strToU8(
      render(templates.flutter.pubspec, { ...vars, PACKAGE_NAME: dartPackageName(run.name) }),
    )
    files['lib/main.dart'] = strToU8(render(templates.flutter.main, vars))
    files['lib/theseus_client.dart'] = strToU8(templates.flutter.client)
    files['README.md'] = strToU8(renderReadme(readmeVars, 'flutter'))
  } else {
    throw new Error(`Export ${exportId}: tier '${tier}' requires a lang, got none`)
  }

  return files
}

/** Build one export's zip, upload it, and flip the row to ready/failed. Never throws — failures are recorded on the row. */
export async function buildBundle(exportId: string): Promise<void> {
  return record('export.assemble', async (span) => {
    span.setAttribute('theseus.export_id', exportId)
    try {
      const files = await assembleFiles(exportId)
      const zipped = zipSync(files, { level: 6 })

      const row = await db.query.modelExports.findFirst({ where: { id: exportId } })
      if (!row) throw new Error(`Export ${exportId} not found`)

      const key = bundleKey(row.runId, exportId)
      await uploadFile(CONSTANTS.BUCKET_MODELS, key, zipped, 'application/zip')
      const checksum = new Bun.CryptoHasher('sha256').update(zipped).digest('hex')

      await db
        .update(modelExports)
        .set({ status: 'ready', bundleKey: key, byteSize: zipped.byteLength, checksum, readyAt: new Date() })
        .where(eq(modelExports.id, exportId))
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      console.error(`[export] assembly failed for ${exportId}:`, e)
      await db
        .update(modelExports)
        .set({ status: 'failed', failedMessage: message })
        .where(eq(modelExports.id, exportId))
    }
  })
}

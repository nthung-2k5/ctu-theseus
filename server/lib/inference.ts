/**
 * Shared inference dispatch/poll logic — used by both the session-cookie
 * routes (routes/inference.ts) and the API-key routes (routes/api-v1.ts),
 * so the two surfaces can't quietly diverge on payload validation or on the
 * pending-row-before-publish ordering that makes `startInferenceResultsConsumer`
 * (lib/microservice.ts) safe (a result can only ever arrive for a row that
 * already exists).
 */

import path from 'node:path'
import { db } from '@server/db'
import { inferenceJobs, type trainingRuns } from '@server/db/schema'
import type { ProjectTask } from '@server/lib/enums'
import type { InferenceInputPayload, InferenceOutput } from '@server/lib/nats'
import { publishInferenceTask, uploadInferenceFile } from '@server/lib/nats'
import { getInferenceInputSpec } from '@server/lib/tasks'

/** A completed batch job's output, stored in inferenceJobs.output — see lib/microservice.ts's startInferenceResultsConsumer. */
export interface BatchInferenceOutput {
  kind: 'batch'
  resultKey: string
  rowCount: number
}

export type PolledInferenceJob =
  | { status: 'pending' }
  | { status: 'success'; output: InferenceOutput }
  | { status: 'batch'; rowCount: number }
  | { status: 'failed'; error: string }

export type DispatchResult = { ok: true; inferenceId: string } | { ok: false; code: 409 | 422; message: string }

/** Parses the multipart `fields` string into a plain JSON object, or null if it isn't one. */
function parseFieldsObject(raw: string): Record<string, unknown> | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null
  return parsed as Record<string, unknown>
}

export interface DispatchInferenceBody {
  file?: File
  fields?: string
  topK?: number
}

/**
 * Validate and dispatch a single-item inference job. `run` must already be
 * confirmed to belong to the caller — this function only checks the run's
 * own state (trained, succeeded) and the payload shape.
 */
export async function dispatchInference(
  run: Pick<typeof trainingRuns.$inferSelect, 'id' | 'status'>,
  task: ProjectTask,
  body: DispatchInferenceBody,
): Promise<DispatchResult> {
  if (run.status !== 'succeeded')
    return { ok: false, code: 409, message: 'No successfully trained model found for this run' }

  const inputSpec = getInferenceInputSpec(task)
  let payload: InferenceInputPayload

  if (inputSpec.kind === 'file') {
    if (!body.file) return { ok: false, code: 422, message: 'This task requires a `file` field' }
    if (inputSpec.accept && !inputSpec.accept.includes(body.file.type)) {
      return { ok: false, code: 422, message: `This task only accepts: ${inputSpec.accept.join(', ')}` }
    }
    const fileBytes = await body.file.bytes()
    const fileExt = path.extname(body.file.name)
    const uploadFilename = `${Bun.randomUUIDv7()}${fileExt}`
    const uploadKey = await uploadInferenceFile(uploadFilename, fileBytes)
    payload = { kind: 'file', uploadKey, uploadFilename }
  } else {
    if (!body.fields) return { ok: false, code: 422, message: 'This task requires a `fields` field' }
    const parsed = parseFieldsObject(body.fields)
    if (!parsed) return { ok: false, code: 422, message: '`fields` must be a JSON-encoded object' }

    if (inputSpec.kind === 'text') {
      const missing = inputSpec.fields.filter((f) => typeof parsed[f] !== 'string' || parsed[f] === '')
      if (missing.length > 0) {
        return { ok: false, code: 422, message: `Missing required field(s): ${missing.join(', ')}` }
      }
      const fields = Object.fromEntries(inputSpec.fields.map((f) => [f, parsed[f] as string]))
      payload = { kind: 'text', fields }
    } else {
      const record: Record<string, string | number> = {}
      for (const [key, value] of Object.entries(parsed)) {
        if (typeof value !== 'string' && typeof value !== 'number') {
          return { ok: false, code: 422, message: `Field '${key}' must be a string or number` }
        }
        record[key] = value
      }
      payload = { kind: 'record', record }
    }
  }

  const inferenceId = Bun.randomUUIDv7()
  // Row must exist before the task is published — startInferenceResultsConsumer
  // only ever UPDATEs an existing row, so publishing first would risk the
  // result arriving before there's anything to update.
  await db.insert(inferenceJobs).values({ id: inferenceId, runId: run.id, status: 'pending' })
  await publishInferenceTask(inferenceId, { inferenceId, runId: run.id, topK: body.topK, payload })
  return { ok: true, inferenceId }
}

/** Validate and dispatch a batch (CSV-of-rows) inference job — text/tabular tasks only. */
export async function dispatchBatchInference(
  run: Pick<typeof trainingRuns.$inferSelect, 'id' | 'status'>,
  task: ProjectTask,
  file: File,
): Promise<DispatchResult> {
  if (run.status !== 'succeeded')
    return { ok: false, code: 409, message: 'No successfully trained model found for this run' }

  // File-backed tasks (vision/audio) need a zip of many files, not a CSV of
  // rows — out of scope (see lib/schema.ts's InferenceTaskSchema doc
  // comment on the `batch` payload kind).
  const inputSpec = getInferenceInputSpec(task)
  if (inputSpec.kind === 'file') {
    return { ok: false, code: 422, message: 'Batch inference is only available for text and tabular tasks' }
  }

  const fileBytes = await file.bytes()
  const uploadFilename = `${Bun.randomUUIDv7()}.csv`
  const uploadKey = await uploadInferenceFile(uploadFilename, fileBytes)

  const inferenceId = Bun.randomUUIDv7()
  await db.insert(inferenceJobs).values({ id: inferenceId, runId: run.id, status: 'pending' })
  await publishInferenceTask(inferenceId, {
    inferenceId,
    runId: run.id,
    payload: { kind: 'batch', uploadKey, uploadFilename },
  })
  return { ok: true, inferenceId }
}

/**
 * Read a dispatched job's current status, scoped to the run it was
 * dispatched for — `null` means no such job exists for this run (either a
 * bad inferenceId, or one belonging to a different run).
 */
export async function pollInferenceJob(runId: string, inferenceId: string): Promise<PolledInferenceJob | null> {
  const job = await db.query.inferenceJobs.findFirst({ where: { id: inferenceId, runId } })
  if (!job) return null
  if (job.status === 'pending') return { status: 'pending' }
  if (job.status === 'failed') return { status: 'failed', error: job.error ?? 'Inference failed' }

  // A batch result has no single InferenceOutput to inline — the caller
  // downloads it via a dedicated route instead of the resultKey (an
  // internal S3 key) being handed back directly here.
  const output = job.output as InferenceOutput | BatchInferenceOutput
  if (output.kind === 'batch') return { status: 'batch', rowCount: output.rowCount }
  return { status: 'success', output }
}

/** The S3 result key for a completed batch job, or null if it isn't one (or isn't done). */
export async function getBatchResultKey(runId: string, inferenceId: string): Promise<string | null> {
  const job = await db.query.inferenceJobs.findFirst({ where: { id: inferenceId, runId } })
  if (job?.status !== 'success') return null
  const output = job.output as InferenceOutput | BatchInferenceOutput
  return output.kind === 'batch' ? output.resultKey : null
}

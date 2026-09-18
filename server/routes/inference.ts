/**
 * Inference routes (session-cookie auth) — dispatch/poll logic itself lives
 * in lib/inference.ts, shared with the API-key surface in routes/api-v1.ts.
 *
 * - POST /api/inference/:runId                              → Dispatch an inference job, returns 202 + inferenceId
 * - POST /api/inference/:runId/batch                        → Dispatch a batch job (CSV of rows) — text/tabular tasks only
 * - GET  /api/inference/:runId/jobs                          → List this run's inference job history (most recent first)
 * - GET  /api/inference/:runId/jobs/:inferenceId             → Poll a dispatched job's status/result
 * - GET  /api/inference/:runId/jobs/:inferenceId/download    → Download a completed batch job's results CSV
 * - POST /api/inference/:runId/warm                          → Preload the run's model into the worker's cache
 *
 * Dispatch is asynchronous (JetStream, via THESEUS_TASKS) rather than a
 * held-open HTTP request — the four `modelType: 'llm'` tasks can generate
 * for longer than any request timeout should reasonably allow, and a
 * cold model load (first request for a run) routinely exceeded the old
 * synchronous path's 10s budget. The dispatch route inserts an
 * `inferenceJobs` row (status 'pending') *before* publishing the task, and
 * a durable gateway consumer (`startInferenceResultsConsumer` in
 * lib/microservice.ts) updates it to success/failed the moment the worker
 * publishes a terminal result on `theseus.inference.result.{inferenceId}` —
 * so the poll route below just reads Postgres, and a result is never lost
 * even if nobody happens to poll before that NATS stream's 1-hour retention
 * window would otherwise have expired it.
 *
 * Body shape depends on the run's task modality (`getInferenceInputSpec`,
 * derived from server/lib/tasks/registry.ts): file-backed tasks
 * (vision/audio) send `file`; text and tabular tasks send `fields`, a
 * JSON-encoded object (multipart form fields can't carry nested objects) —
 * one entry per Ludwig input column for text tasks (e.g. `context` +
 * `question` for question_answering), or the whole row for tabular tasks.
 */

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { dispatchBatchInference, dispatchInference, getBatchResultKey, pollInferenceJob } from '@server/lib/inference'
import { publishInferenceWarm } from '@server/lib/nats'
import { getDownloadUrl } from '@server/lib/storage'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

export const inferenceRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)
  /* ── Dispatch an inference job ── */
  .post(
    '/inference/:runId',
    async ({ run, body }) => {
      if (!run.project) return status(404, 'Run not found')
      const result = await dispatchInference(run, run.project.task, body)
      if (!result.ok) return status(result.code, result.message)
      return status(202, { inferenceId: result.inferenceId })
    },
    {
      runBelongToUser: true,
      body: t.Object({
        file: t.Optional(t.File({ maxSize: '25m' })),
        /** JSON-encoded object — one value per input field (text tasks) or the whole row (tabular tasks). */
        fields: t.Optional(t.String()),
        topK: t.Optional(t.Numeric({ minimum: 1, maximum: 1000 })),
      }),
    },
  )
  /* ── Dispatch a batch job: one CSV row per prediction, scored in a single Ludwig predict call ── */
  .post(
    '/inference/:runId/batch',
    async ({ run, body }) => {
      if (!run.project) return status(404, 'Run not found')
      const result = await dispatchBatchInference(run, run.project.task, body.file)
      if (!result.ok) return status(result.code, result.message)
      return status(202, { inferenceId: result.inferenceId })
    },
    {
      runBelongToUser: true,
      body: t.Object({ file: t.File({ maxSize: '25m' }) }),
    },
  )
  /* ── List this run's inference job history, most recent first ── */
  .get(
    '/inference/:runId/jobs',
    async ({ run }) => {
      const jobs = await db.query.inferenceJobs.findMany({
        where: { runId: run.id },
        orderBy: { createdAt: 'desc' },
        limit: 50,
      })
      return { jobs }
    },
    { runBelongToUser: true },
  )
  /* ── Poll a dispatched job ── */
  .get(
    '/inference/:runId/jobs/:inferenceId',
    async ({ params }) => {
      // Scoped by runId in the same query — an inferenceId from another
      // user's run simply won't match, so there's no separate ownership
      // cross-check to get wrong here (unlike the old NATS-backed version,
      // where THESEUS_INFERENCE_RESULTS was keyed on inferenceId alone).
      const job = await pollInferenceJob(params.runId, params.inferenceId)
      if (!job) return status(404, 'Inference job not found for this run')
      return job
    },
    {
      runBelongToUser: true,
      // runBelongToUser's own `params` schema only declares `runId`; without
      // redeclaring the full set here, Elysia validates against that
      // narrower shape and rejects `inferenceId` as an unexpected property
      // before the handler ever runs (see classes.ts for the same pattern).
      params: t.Object({ runId: t.String({ format: 'uuid' }), inferenceId: t.String({ format: 'uuid' }) }),
    },
  )
  /* ── Download a completed batch job's results CSV via S3 presigned URL ── */
  .get(
    '/inference/:runId/jobs/:inferenceId/download',
    async ({ params }) => {
      const resultKey = await getBatchResultKey(params.runId, params.inferenceId)
      if (!resultKey) return status(404, 'Inference job not found for this run, or has no downloadable result')
      const url = await getDownloadUrl(CONSTANTS.BUCKET_MODELS, resultKey)
      return new Response(null, { status: 302, headers: { Location: url } })
    },
    {
      runBelongToUser: true,
      params: t.Object({ runId: t.String({ format: 'uuid' }), inferenceId: t.String({ format: 'uuid' }) }),
    },
  )
  /* ── Preload a run's model before the user submits their first request ── */
  .post(
    '/inference/:runId/warm',
    ({ run }) => {
      if (run.status !== 'succeeded') return status(409, 'No successfully trained model found for this run')
      publishInferenceWarm(run.id)
      return status(202)
    },
    { runBelongToUser: true },
  )

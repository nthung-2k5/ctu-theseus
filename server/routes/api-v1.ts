/**
 * Hosted prediction API (bearer-token auth via `Authorization: Bearer <key>`
 * — see the `apiKeyAuth`/`apiKeyRunBelongToUser` macros in routes/auth.ts
 * and routes/apiKeys.ts for issuing a key). The only surface in this app
 * callable without a browser session cookie, which is why every route here
 * is rate-limited per key (see `apiKeyAuth`) — none of the session-cookie
 * routes need that, since a browser session isn't something a script can
 * cheaply mint in bulk the way an API key request can be retried.
 *
 * Dispatch/poll logic is shared with routes/inference.ts via
 * lib/inference.ts — this file is deliberately thin so the two auth
 * surfaces can't drift on validation or on the pending-row-before-publish
 * ordering that makes result persistence safe.
 *
 * - POST /api/v1/predict/:runId                            → Dispatch, returns 202 + inferenceId (same as the session route)
 * - POST /api/v1/predict/:runId/sync                       → Dispatch and wait up to ~25s for the result inline
 * - POST /api/v1/predict/:runId/batch                      → Dispatch a batch job (CSV of rows) — text/tabular tasks only
 * - GET  /api/v1/predict/:runId/jobs/:inferenceId           → Poll a dispatched job
 * - GET  /api/v1/predict/:runId/jobs/:inferenceId/download  → Download a completed batch job's results CSV
 */

import CONSTANTS from '@schema/constants.json'
import {
  dispatchBatchInference,
  dispatchInference,
  getBatchResultKey,
  type PolledInferenceJob,
  pollInferenceJob,
} from '@server/lib/inference'
import { getDownloadUrl } from '@server/lib/storage'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/** Bounded-wait budget for the /sync variant — comfortably under a typical HTTP client's own default timeout. */
const SYNC_MAX_WAIT_MS = 25_000
const SYNC_POLL_INTERVAL_MS = 500

async function waitForResult(runId: string, inferenceId: string): Promise<PolledInferenceJob | null> {
  const deadline = Date.now() + SYNC_MAX_WAIT_MS
  while (Date.now() < deadline) {
    const job = await pollInferenceJob(runId, inferenceId)
    if (job && job.status !== 'pending') return job
    await new Promise((resolve) => setTimeout(resolve, SYNC_POLL_INTERVAL_MS))
  }
  return pollInferenceJob(runId, inferenceId)
}

export const apiV1Routes = new Elysia({ prefix: '/api/v1' })
  .use(betterAuth)
  /* ── Dispatch an inference job ── */
  .post(
    '/predict/:runId',
    async ({ run, body }) => {
      if (!run.project) return status(404, 'Run not found')
      const result = await dispatchInference(run, run.project.task, body)
      if (!result.ok) return status(result.code, result.message)
      return status(202, { inferenceId: result.inferenceId })
    },
    {
      apiKeyRunBelongToUser: true,
      body: t.Object({
        file: t.Optional(t.File({ maxSize: '25m' })),
        fields: t.Optional(t.String()),
        topK: t.Optional(t.Numeric({ minimum: 1, maximum: 1000 })),
      }),
    },
  )
  /* ── Dispatch and wait up to SYNC_MAX_WAIT_MS for a terminal result before falling back to the async shape ── */
  .post(
    '/predict/:runId/sync',
    async ({ run, body }) => {
      if (!run.project) return status(404, 'Run not found')
      const result = await dispatchInference(run, run.project.task, body)
      if (!result.ok) return status(result.code, result.message)

      const job = await waitForResult(run.id, result.inferenceId)
      if (!job || job.status === 'pending') {
        // Didn't finish in time — hand back the id so the caller can poll
        // the async route instead of holding the connection open forever.
        return status(202, { inferenceId: result.inferenceId, status: 'pending' as const })
      }
      return { inferenceId: result.inferenceId, ...job }
    },
    {
      apiKeyRunBelongToUser: true,
      body: t.Object({
        file: t.Optional(t.File({ maxSize: '25m' })),
        fields: t.Optional(t.String()),
        topK: t.Optional(t.Numeric({ minimum: 1, maximum: 1000 })),
      }),
    },
  )
  /* ── Dispatch a batch job: one CSV row per prediction, scored in a single Ludwig predict call ── */
  .post(
    '/predict/:runId/batch',
    async ({ run, body }) => {
      if (!run.project) return status(404, 'Run not found')
      const result = await dispatchBatchInference(run, run.project.task, body.file)
      if (!result.ok) return status(result.code, result.message)
      return status(202, { inferenceId: result.inferenceId })
    },
    {
      apiKeyRunBelongToUser: true,
      body: t.Object({ file: t.File({ maxSize: '25m' }) }),
    },
  )
  /* ── Poll a dispatched job ── */
  .get(
    '/predict/:runId/jobs/:inferenceId',
    async ({ params }) => {
      const job = await pollInferenceJob(params.runId, params.inferenceId)
      if (!job) return status(404, 'Inference job not found for this run')
      return job
    },
    {
      apiKeyRunBelongToUser: true,
      params: t.Object({ runId: t.String({ format: 'uuid' }), inferenceId: t.String({ format: 'uuid' }) }),
    },
  )
  /* ── Download a completed batch job's results CSV via S3 presigned URL ── */
  .get(
    '/predict/:runId/jobs/:inferenceId/download',
    async ({ params }) => {
      const resultKey = await getBatchResultKey(params.runId, params.inferenceId)
      if (!resultKey) return status(404, 'Inference job not found for this run, or has no downloadable result')
      const url = await getDownloadUrl(CONSTANTS.BUCKET_MODELS, resultKey)
      return new Response(null, { status: 302, headers: { Location: url } })
    },
    {
      apiKeyRunBelongToUser: true,
      params: t.Object({ runId: t.String({ format: 'uuid' }), inferenceId: t.String({ format: 'uuid' }) }),
    },
  )

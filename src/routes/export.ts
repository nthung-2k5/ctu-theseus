/**
 * Export routes – handles model export via NATS and provides download links.
 */

import CONSTANTS from '@schema/constants.json'
import { dispatchExport } from '@server/lib/microservice'
import { fileExists, getExportDownloadUrl } from '@server/lib/storage'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/* ------------------------------------------------------------------ */
/*  Routes                                                            */
/* ------------------------------------------------------------------ */

export const exportRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)

  /* ── Dispatch a model export job for a succeeded run ── */
  .post(
    '/runs/:runId/export',
    async ({ run, body }) => {
      if (run.status !== 'succeeded') return status(409, `Run is not succeeded (status: ${run.status})`)
      const jobId = await dispatchExport(run.id, body.format)
      return status(202, { jobId })
    },
    {
      runBelongToUser: true,
      body: t.Object({ format: t.UnionEnum(['onnx', 'torchscript']) }),
    },
  )

  /* ── Check whether an export artifact exists yet ── */
  .get(
    '/runs/:runId/export/:format',
    async ({ run, params }) => {
      const exists = await fileExists(CONSTANTS.BUCKET_MODELS, `${run.id}/model.${params.format}`)
      return { ready: exists !== null }
    },
    { runBelongToUser: true, params: t.Object({ runId: t.String(), format: t.String() }) },
  )

  /* ── Download exported model file via S3 presigned URL ── */
  .get(
    '/runs/:runId/download/:format',
    async ({ run, params }) => {
      if (run.status !== 'succeeded') {
        return status(404, 'No successfully trained model found')
      }

      // Generate a presigned download URL from S3 and redirect to it.
      const downloadUrl = await getExportDownloadUrl(run.id, params.format)
      return new Response(null, {
        status: 302,
        headers: {
          Location: downloadUrl,
        },
      })
    },
    {
      runBelongToUser: true,
      params: t.Object({
        runId: t.String(),
        format: t.String(),
      }),
    },
  )

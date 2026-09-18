/**
 * Export routes – three tiers (model/devkit/app) assembled as a zip bundle,
 * under /runs/:runId/exports.
 *
 * Two-phase flow for a new export:
 *   1. POST /runs/:runId/exports — if the converted model artifact already
 *      exists (a prior export in another tier/lang already triggered
 *      conversion), skip straight to 'assembling'; otherwise dispatch a
 *      conversion task to the worker and sit in 'converting'.
 *   2. The worker's completion event flips 'converting' -> 'assembling' and
 *      enqueues zip assembly (lib/microservice.ts's `export` case). Events
 *      can be missed, so GET /exports/:exportId also lazily reconciles.
 */

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { modelExports } from '@server/db/schema'
import { AppTargets, DevkitLangs, ExportFormats, ExportLangs, ExportTiers } from '@server/lib/enums'
import { enqueueAssembly } from '@server/lib/export/queue'
import { dispatchExport } from '@server/lib/microservice'
import { exportKey, fileExists, getBundleDownloadUrl } from '@server/lib/storage'
import { and, eq } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/** How long an export may sit in 'converting' before the worker is presumed lost. */
const CONVERSION_TIMEOUT_MS = 15 * 60 * 1000

/**
 * If a 'converting' export's artifact now exists (event was missed) or its
 * conversion has clearly stalled, resolve it. Called on every GET so the
 * client's poll loop is also the recovery mechanism — no cron needed.
 */
async function reconcileConverting(exportRow: typeof modelExports.$inferSelect): Promise<void> {
  if (exportRow.status !== 'converting') return

  const artifactExists = await fileExists(CONSTANTS.BUCKET_MODELS, exportKey(exportRow.runId, exportRow.format))
  if (artifactExists) {
    const promoted = await db
      .update(modelExports)
      .set({ status: 'assembling' })
      .where(and(eq(modelExports.id, exportRow.id), eq(modelExports.status, 'converting')))
      .returning({ id: modelExports.id })
    if (promoted.length > 0) enqueueAssembly(exportRow.id)
    return
  }

  if (Date.now() - exportRow.createdAt.getTime() > CONVERSION_TIMEOUT_MS) {
    await db
      .update(modelExports)
      .set({ status: 'failed', failedMessage: 'Model conversion timed out' })
      .where(and(eq(modelExports.id, exportRow.id), eq(modelExports.status, 'converting')))
  }
}

export const exportRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)

  /* ── Dispatch a new export (any tier) ── */
  .post(
    '/runs/:runId/exports',
    async ({ run, body }) => {
      if (run.status !== 'succeeded') return status(409, `Run is not succeeded (status: ${run.status})`)
      if (!run.project) return status(404, 'Run not found')

      if (body.tier !== 'model') {
        if (body.format !== 'onnx') return status(400, `tier '${body.tier}' only supports format 'onnx'`)
        if (!body.lang) return status(400, `tier '${body.tier}' requires a lang`)
        const validLangs: readonly string[] = body.tier === 'devkit' ? DevkitLangs : AppTargets
        if (!validLangs.includes(body.lang)) {
          return status(400, `tier '${body.tier}' requires lang to be one of: ${validLangs.join(', ')}`)
        }
      }

      const [row] = await db
        .insert(modelExports)
        .values({
          runId: run.id,
          userId: run.project.userId,
          tier: body.tier,
          format: body.format,
          lang: body.tier === 'model' ? null : body.lang,
          status: 'pending',
        })
        .returning()

      const artifactExists = await fileExists(CONSTANTS.BUCKET_MODELS, exportKey(run.id, body.format))
      if (artifactExists) {
        await db.update(modelExports).set({ status: 'assembling' }).where(eq(modelExports.id, row.id))
        enqueueAssembly(row.id)
      } else {
        const jobId = await dispatchExport(run.id, body.format)
        await db
          .update(modelExports)
          .set({ status: 'converting', conversionJobId: jobId })
          .where(eq(modelExports.id, row.id))
      }

      return status(202, { exportId: row.id })
    },
    {
      runBelongToUser: true,
      body: t.Object({
        tier: t.UnionEnum(ExportTiers),
        format: t.UnionEnum(ExportFormats),
        lang: t.Optional(t.UnionEnum(ExportLangs)),
      }),
    },
  )

  /* ── List a run's exports ── */
  .get(
    '/runs/:runId/exports',
    async ({ run }) => {
      const exports = await db.query.modelExports.findMany({
        where: { runId: run.id },
        orderBy: { createdAt: 'desc' },
      })
      return { exports }
    },
    { runBelongToUser: true },
  )

  /* ── Get one export's status, reconciling a stalled 'converting' state first ── */
  .get(
    '/exports/:exportId',
    async ({ modelExport, params }) => {
      await reconcileConverting(modelExport)
      const current = await db.query.modelExports.findFirst({ where: { id: params.exportId } })
      return { export: current ?? modelExport }
    },
    { exportBelongToUser: true },
  )

  /* ── Download a ready export bundle via S3 presigned URL ── */
  .get(
    '/exports/:exportId/download',
    async ({ modelExport }) => {
      if (modelExport.status !== 'ready' || !modelExport.bundleKey) {
        return status(404, 'Export is not ready for download')
      }
      const downloadUrl = await getBundleDownloadUrl(modelExport.runId, modelExport.id)
      return new Response(null, { status: 302, headers: { Location: downloadUrl } })
    },
    { exportBelongToUser: true },
  )

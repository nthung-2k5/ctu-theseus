import { auth } from '@server/auth'
import { db } from '@server/db'
import { apiKeys } from '@server/db/schema'
import { hashApiKey } from '@server/lib/apiKeys'
import { checkRateLimit } from '@server/lib/rateLimit'
import { eq } from 'drizzle-orm'
import Elysia, { t } from 'elysia'

/** Requests per key per window for the hosted prediction API — see `apiKeyAuth` below. */
const API_KEY_RATE_LIMIT_MAX = 60
const API_KEY_RATE_LIMIT_WINDOW_MS = 60 * 1000

export const betterAuth = new Elysia({ name: 'better-auth' })
  .mount(auth.handler)
  .macro('auth', {
    async resolve({ status, request: { headers } }) {
      const session = await auth.api.getSession({
        headers,
      })
      if (!session) return status(401)
      return {
        user: session.user,
        session: session.session,
      }
    },
  })
  /**
   * Bearer-token auth for the hosted prediction API (routes/api-v1.ts) —
   * the first surface in this app not behind a session cookie, which is
   * exactly why it needs its own rate limit (`checkRateLimit`, keyed by the
   * key's hash — never the raw secret — so a request is throttled before
   * ever touching Postgres, whether or not the key turns out to be valid).
   */
  .macro('apiKeyAuth', {
    async resolve({ status, request: { headers } }) {
      const authHeader = headers.get('authorization')
      const rawKey = authHeader?.startsWith('Bearer ') ? authHeader.slice('Bearer '.length) : null
      if (!rawKey) return status(401, 'Missing API key — send it as `Authorization: Bearer <key>`')

      const keyHash = hashApiKey(rawKey)
      if (!checkRateLimit(keyHash, API_KEY_RATE_LIMIT_MAX, API_KEY_RATE_LIMIT_WINDOW_MS)) {
        return status(429, 'Rate limit exceeded — try again shortly')
      }

      const apiKey = await db.query.apiKeys.findFirst({ where: { keyHash, revokedAt: { isNull: true } } })
      if (!apiKey) return status(401, 'Invalid or revoked API key')

      // Best-effort — a failed update must not block the actual request.
      db.update(apiKeys).set({ lastUsedAt: new Date() }).where(eq(apiKeys.id, apiKey.id)).catch(() => {})

      return { apiKeyUserId: apiKey.userId }
    },
  })
  .macro('apiKeyRunBelongToUser', {
    apiKeyAuth: true,
    params: t.Object({
      runId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, apiKeyUserId }) {
      const run = await db.query.trainingRuns.findFirst({
        where: { id: params.runId },
        with: { project: true },
      })
      if (!run?.project) return status(404, 'Training run not found')
      if (run.project.userId !== apiKeyUserId) return status(403, 'Unauthorized')
      return { run }
    },
  })
  .macro('projectBelongToUser', {
    auth: true,
    params: t.Object({
      projectId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const project = await db.query.projects.findFirst({
        where: {
          id: params.projectId,
        },
      })
      if (!project) return status(404, 'Project not found')
      if (project.userId !== user.id) return status(403, 'Unauthorized')
      return {
        project,
      }
    },
  })
  .macro('versionBelongToUser', {
    auth: true,
    params: t.Object({
      versionId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const version = await db.query.datasetVersions.findFirst({
        where: { id: params.versionId },
        with: { dataset: { with: { project: true } } },
      })
      if (!version?.dataset.project) return status(404, 'Version not found')
      if (version.dataset.project.userId !== user.id) return status(403, 'Unauthorized')
      return { version }
    },
  })
  .macro('sweepBelongToUser', {
    auth: true,
    params: t.Object({
      sweepId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const sweep = await db.query.sweeps.findFirst({
        where: { id: params.sweepId },
        with: { project: true },
      })
      if (!sweep?.project) return status(404, 'Sweep not found')
      if (sweep.project.userId !== user.id) return status(403, 'Unauthorized')
      return { sweep }
    },
  })
  .macro('runBelongToUser', {
    auth: true,
    params: t.Object({
      runId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const run = await db.query.trainingRuns.findFirst({
        where: { id: params.runId },
        with: { project: true },
      })
      if (!run?.project) return status(404, 'Training run not found')
      if (run.project.userId !== user.id) return status(403, 'Unauthorized')
      return { run }
    },
  })
  .macro('itemBelongToUser', {
    auth: true,
    params: t.Object({
      itemId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const item = await db.query.datasetItems.findFirst({
        where: { id: params.itemId },
        with: { dataset: { with: { project: true } } },
      })
      if (!item?.dataset.project) return status(404, 'Item not found')
      if (item.dataset.project.userId !== user.id) return status(403, 'Unauthorized')
      return { item }
    },
  })
  .macro('draftBelongToUser', {
    auth: true,
    params: t.Object({
      projectId: t.String({ format: 'uuid' }),
    }),
    /**
     * Resolves a project's mutable draft version alongside the project itself.
     *
     * Five routes used to do `projectBelongToUser` and then repeat the same
     * `findFirst({ datasetId, versionTag: isNull })` + `404 'Draft dataset not
     * found'`. Joining through to the project here also makes it one query
     * instead of two.
     */
    async resolve({ status, params, user }) {
      const draft = await db.query.datasetVersions.findFirst({
        where: { datasetId: params.projectId, versionTag: { isNull: true } },
        with: { dataset: { with: { project: true } } },
      })
      if (!draft?.dataset.project) return status(404, 'Draft dataset not found')
      if (draft.dataset.project.userId !== user.id) return status(403, 'Unauthorized')
      return { draft, project: draft.dataset.project }
    },
  })
  .macro('annotationBelongToUser', {
    auth: true,
    params: t.Object({
      annotationId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const annotation = await db.query.annotations.findFirst({
        where: { id: params.annotationId },
        with: { item: { with: { dataset: { with: { project: true } } } } },
      })
      if (!annotation?.item.dataset.project) return status(404, 'Annotation not found')
      if (annotation.item.dataset.project.userId !== user.id) return status(403, 'Unauthorized')
      return { annotation }
    },
  })
  .macro('exportBelongToUser', {
    auth: true,
    params: t.Object({
      exportId: t.String({ format: 'uuid' }),
    }),
    async resolve({ status, params, user }) {
      const modelExport = await db.query.modelExports.findFirst({
        where: { id: params.exportId },
        with: { run: { with: { project: true } } },
      })
      if (!modelExport?.run.project) return status(404, 'Export not found')
      if (modelExport.run.project.userId !== user.id) return status(403, 'Unauthorized')
      return { modelExport }
    },
  })

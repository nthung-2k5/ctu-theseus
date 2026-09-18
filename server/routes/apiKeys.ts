/**
 * API key management (session-cookie auth) — issuing/listing/revoking the
 * bearer credentials POST /api/v1/predict/:runId accepts (see
 * routes/api-v1.ts and the `apiKeyAuth` macro in routes/auth.ts).
 *
 * - POST   /api/keys           → Create a key, returns the raw key ONCE — never stored, never shown again
 * - GET    /api/keys           → List the caller's keys (prefix + metadata only, never the raw key)
 * - DELETE /api/keys/:keyId    → Revoke a key (soft — keeps the audit trail of what it was)
 */

import { db } from '@server/db'
import { apiKeys } from '@server/db/schema'
import { generateApiKey } from '@server/lib/apiKeys'
import { and, eq, isNull } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

export const apiKeyRoutes = new Elysia({ prefix: '/api/keys' })
  .use(betterAuth)
  /* ── Create a new API key ── */
  .post(
    '/',
    async ({ user, body }) => {
      const { rawKey, keyHash, keyPrefix } = generateApiKey()
      const [key] = await db
        .insert(apiKeys)
        .values({ userId: user.id, name: body.name, keyHash, keyPrefix })
        .returning({ id: apiKeys.id, name: apiKeys.name, keyPrefix: apiKeys.keyPrefix, createdAt: apiKeys.createdAt })

      // The only point in this key's lifetime the raw value is ever
      // available — it is not derivable from keyHash, so losing this
      // response means generating a new key.
      return status(201, { ...key, key: rawKey })
    },
    {
      auth: true,
      body: t.Object({ name: t.String({ minLength: 1, maxLength: 100 }) }),
    },
  )
  /* ── List the caller's keys — never the raw value, only what was shown at creation ── */
  .get(
    '/',
    async ({ user }) => {
      const keys = await db.query.apiKeys.findMany({
        where: { userId: user.id },
        columns: { id: true, name: true, keyPrefix: true, lastUsedAt: true, createdAt: true, revokedAt: true },
        orderBy: { createdAt: 'desc' },
      })
      return { keys }
    },
    { auth: true },
  )
  /* ── Revoke a key ── */
  .delete(
    '/:keyId',
    async ({ user, params }) => {
      const [revoked] = await db
        .update(apiKeys)
        .set({ revokedAt: new Date() })
        .where(and(eq(apiKeys.id, params.keyId), eq(apiKeys.userId, user.id), isNull(apiKeys.revokedAt)))
        .returning({ id: apiKeys.id })
      if (!revoked) return status(404, 'API key not found')
      return status(204)
    },
    {
      auth: true,
      params: t.Object({ keyId: t.String({ format: 'uuid' }) }),
    },
  )

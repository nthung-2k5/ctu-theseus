/**
 * Shared harness for route-level integration tests (`*.integration.test.ts`).
 *
 * These tests hit real Elysia routes composed with real Postgres/better-auth
 * — no mocking of `db` or `auth`, since neither has an injection seam today.
 * They need a real, migrated Postgres reachable at `CTU_THESEUS_DB_URI` (the
 * same one `aspire run` provisions) and are NOT part of the default `bun
 * test`/CI run: `server/bunfig.toml`'s `[test] pathIgnorePatterns` excludes
 * `*.integration.test.ts` globally, for *any* invocation style. Run them
 * explicitly with `bun run test:integration`, which points at
 * `bunfig.integration.toml` (no ignore pattern) instead, once a
 * dev Postgres is up and migrated.
 *
 * They never exercise a route that touches S3 (file upload, project/version
 * deletion) — no S3-compatible store is assumed to be running. Every test
 * item is created inline (text/tabular) via POST /items, which never writes
 * `storageUrl`, so cleanup here never needs to delete an S3 object either.
 */

import { auth } from '@server/auth'
import { db } from '@server/db'
import { datasetVersionItems, projects, users } from '@server/db/schema'
import { betterAuth } from '@server/routes/auth'
import { classRoutes } from '@server/routes/classes'
import { datasetRoutes } from '@server/routes/datasets'
import { projectRoutes } from '@server/routes/projects'
import { eq, inArray } from 'drizzle-orm'
import { Elysia } from 'elysia'

/** Composed app under test — mirrors server/index.ts's route wiring minus infra bootstrapping (NATS/S3/reapers). */
export const testApp = new Elysia()
  .mount(auth.handler)
  .use(betterAuth)
  .use(projectRoutes)
  .use(classRoutes)
  .use(datasetRoutes)

/** A `name=value` Cookie header string extracted from a Set-Cookie response header, stripped of attributes. */
function extractCookie(response: Response): string {
  const headers = response.headers as Headers & { getSetCookie?: () => string[] }
  const single = response.headers.get('set-cookie')
  const setCookies = headers.getSetCookie?.() ?? (single ? [single] : [])
  return setCookies
    .map((c) => c.split(';')[0]?.trim())
    .filter((c): c is string => !!c)
    .join('; ')
}

export type TestUser = { userId: string; cookie: string }

/** Signs up a throwaway user via the real better-auth email/password flow and returns its session cookie. */
export async function createTestUser(): Promise<TestUser> {
  const email = `itest-${crypto.randomUUID()}@example.com`
  const response = await testApp.handle(
    new Request('http://localhost/api/auth/sign-up/email', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email, password: 'Test1234!', name: 'Integration Test User' }),
    }),
  )
  if (response.status !== 200) {
    throw new Error(`createTestUser: sign-up failed (${response.status}): ${await response.text()}`)
  }
  const cookie = extractCookie(response)
  if (!cookie) throw new Error('createTestUser: sign-up response carried no session cookie')
  const body = (await response.json()) as { user: { id: string } }
  return { userId: body.user.id, cookie }
}

/** Hard-deletes a test user — cascades (users -> projects -> datasets -> ... ) clean up everything it owns. */
export async function deleteTestUser(userId: string): Promise<void> {
  await db.delete(users).where(eq(users.id, userId))
}

/**
 * Deletes a test project. Mirrors DELETE /api/projects/:projectId (routes/projects.ts)
 * minus its `cleanupProjectStorage` S3 call — every item in these tests is
 * inline (no `storageUrl`), so there's nothing in S3 to clean up. Pre-deleting
 * `dataset_version_items` avoids the ON DELETE RESTRICT FK on its `itemId`
 * column tripping during the cascade, exactly as the real route does.
 */
export async function cleanupTestProject(projectId: string): Promise<void> {
  const versions = await db.query.datasetVersions.findMany({
    where: { datasetId: projectId },
    columns: { id: true },
  })
  if (versions.length > 0) {
    await db.delete(datasetVersionItems).where(
      inArray(
        datasetVersionItems.versionId,
        versions.map((v) => v.id),
      ),
    )
  }
  await db.delete(projects).where(eq(projects.id, projectId))
}

/** Builds an authenticated JSON Request against the test app. */
export function authedRequest(method: string, path: string, cookie: string, body?: unknown): Request {
  return new Request(`http://localhost${path}`, {
    method,
    headers: {
      'content-type': 'application/json',
      cookie,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
}

/** Convenience wrapper: fires the request through `testApp` and parses the JSON body (or returns null for empty/204 responses). */
export async function callApi(
  method: string,
  path: string,
  cookie: string,
  body?: unknown,
): Promise<{ status: number; body: unknown }> {
  const response = await testApp.handle(authedRequest(method, path, cookie, body))
  const status = response.status
  if (status === 204 || response.headers.get('content-length') === '0') return { status, body: null }
  const text = await response.text()
  if (!text) return { status, body: null }
  try {
    return { status, body: JSON.parse(text) }
  } catch {
    return { status, body: text }
  }
}

/** Finds the draft (mutable, versionTag = null) version id for a project's dataset. */
export async function getDraftVersionId(projectId: string): Promise<string> {
  const draft = await db.query.datasetVersions.findFirst({
    where: { datasetId: projectId, versionTag: { isNull: true } },
    columns: { id: true },
  })
  if (!draft) throw new Error(`getDraftVersionId: no draft version found for project ${projectId}`)
  return draft.id
}

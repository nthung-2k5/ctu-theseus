import { db } from '@server/db'
import { datasets, datasetVersionItems, datasetVersions, projects, trainingRuns } from '@server/db/schema'
import { cleanupProjectStorage } from '@server/lib/cleanup'
import { ProjectTasks } from '@server/lib/enums'
import { getTaskDescriptor, taskToModality } from '@server/lib/tasks'
import { count, eq, inArray } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

type SplitCounts = { train: number; validation: number; test: number }

const EMPTY_SPLIT_COUNTS: SplitCounts = { train: 0, validation: 0, test: 0 }

/**
 * Attach per-split membership counts (and their total) to a version.
 *
 * `itemCount` is derived here rather than read off the column: the column is
 * only written when a snapshot is built, so it is null for the draft — which
 * is exactly the version the UI shows a live count for.
 */
function withCounts<T extends { id: string }>(version: T, counts: Map<string, SplitCounts>) {
  const splitCounts = counts.get(version.id) ?? EMPTY_SPLIT_COUNTS
  return {
    ...version,
    splitCounts,
    itemCount: splitCounts.train + splitCounts.validation + splitCounts.test,
  }
}

export const projectRoutes = new Elysia({ prefix: '/api/projects' })
  .use(betterAuth)
  /* ── List all projects for the authenticated user ── */
  .get(
    '/',
    async ({ user }) => {
      const rows = await db.query.projects.findMany({
        where: {
          userId: user.id,
        },
        orderBy: {
          createdAt: 'desc',
        },
        columns: {
          id: true,
          name: true,
          description: true,
          task: true,
          createdAt: true,
        },
        with: {
          draftDataset: {
            columns: { modality: true },
          },
        },
      })

      return { projects: rows }
    },
    { auth: true },
  )
  /* ── Get single project with aggregated counts ── */
  .get(
    '/:projectId',
    async ({ project }) => {
      const [versionCount, runCount, datasetRow, splitRows] = await Promise.all([
        db.$count(datasetVersions, eq(datasetVersions.datasetId, project.id)),
        db.$count(trainingRuns, eq(trainingRuns.projectId, project.id)),
        // Deliberately NOT `with: { items: true }`. That loaded the full
        // membership set for the draft *and* every snapshot on the route the
        // frontend hits on every project navigation — hundreds of thousands of
        // rows for a large pool — when the only thing any consumer reads off
        // it is counts (the sidebar badge, the stat tile, the split bar).
        db.query.datasets.findFirst({
          where: { projectId: project.id },
          with: { draft: true, versions: true, classes: true },
        }),
        // One grouped query, at most three rows per version.
        db
          .select({
            versionId: datasetVersionItems.versionId,
            splitType: datasetVersionItems.splitType,
            n: count(),
          })
          .from(datasetVersionItems)
          .innerJoin(datasetVersions, eq(datasetVersions.id, datasetVersionItems.versionId))
          .where(eq(datasetVersions.datasetId, project.id))
          .groupBy(datasetVersionItems.versionId, datasetVersionItems.splitType),
      ])

      const countsByVersion = new Map<string, SplitCounts>()
      for (const row of splitRows) {
        const entry = countsByVersion.get(row.versionId) ?? { train: 0, validation: 0, test: 0 }
        entry[row.splitType] = row.n
        countsByVersion.set(row.versionId, entry)
      }

      const dataset = datasetRow
        ? {
            ...datasetRow,
            draft: datasetRow.draft ? withCounts(datasetRow.draft, countsByVersion) : null,
            versions: datasetRow.versions.map((v) => withCounts(v, countsByVersion)),
          }
        : null

      return { project: { ...project, runCount, versionCount, dataset } }
    },
    {
      projectBelongToUser: true,
    },
  )
  /* ── Create a new project (also creates the 1:1 dataset + draft version + 3 splits) ── */
  .post(
    '/',
    async ({ body, user }) => {
      const descriptor = getTaskDescriptor(body.task)
      if (descriptor.backend !== 'ludwig') {
        return status(422, `Task "${body.task}" is not yet trainable (${descriptor.status}).`)
      }

      // Derive modality from task
      const modality = taskToModality(body.task)

      // All three inserts must succeed together — a project with no dataset
      // (or a dataset with no draft version) is unusable, so a failure
      // partway through must not leave any of them behind.
      const project = await db.transaction(async (tx) => {
        const [project] = await tx
          .insert(projects)
          .values({
            ...body,
            userId: user.id,
          })
          .returning()

        // Create the 1:1 dataset (PK = project.id)
        await tx.insert(datasets).values({
          projectId: project.id,
          modality,
        })

        // Create the draft version (versionTag = null). Splits are just a
        // column on each item's version-membership row now, so there's
        // nothing else to pre-create here.
        await tx.insert(datasetVersions).values({
          datasetId: project.id, // datasetId references datasets.projectId
        })

        return project
      })

      return { project }
    },
    {
      body: t.Object({
        name: t.String({ minLength: 1 }),
        description: t.Nullable(t.String()),
        task: t.UnionEnum(ProjectTasks),
      }),
      auth: true,
    },
  )
  /* ── Update project ── */
  .patch(
    '/:projectId',
    async ({ project, body }) => {
      const [updated] = await db.update(projects).set(body).where(eq(projects.id, project.id)).returning()
      return { project: updated }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        name: t.Optional(t.String({ minLength: 1 })),
        description: t.MaybeEmpty(t.String()),
      }),
    },
  )
  /* ── Delete project (cascades to dataset, versions, items, runs, etc.; S3 objects cleaned up best-effort) ── */
  .delete(
    '/:projectId',
    async ({ project }) => {
      // Gather + delete S3 objects before the DB cascade removes the rows
      // that reference their keys — cleanupProjectStorage queries them.
      await cleanupProjectStorage(project.id)
      var versions = await db.query.datasetVersions.findMany({
        where: { datasetId: project.id },
        columns: { id: true },
      })
      await db.delete(datasetVersionItems).where(inArray(datasetVersionItems.versionId, versions.map((v) => v.id)))
      await db.delete(projects).where(eq(projects.id, project.id))
      return status(204)
    },
    { projectBelongToUser: true },
  )

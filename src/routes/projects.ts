import { db } from '@server/db'
import { datasetSplits, datasets, datasetVersions, projects, trainingRuns } from '@server/db/schema'
import { ProjectTasks } from '@server/lib/enums'
import { taskToModality } from '@server/lib/helpers'
import { and, eq } from 'drizzle-orm'
import { Elysia, NotFoundError, status, t } from 'elysia'
import { betterAuth } from './auth'

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
      const [versionCount, runCount] = await Promise.all([
        db.$count(datasetVersions, eq(datasetVersions.datasetId, project.id)),
        db.$count(trainingRuns, eq(trainingRuns.projectId, project.id)),
      ])

      return { project: { ...project, runCount, versionCount } }
    },
    {
      projectBelongToUser: true,
    },
  )
  /* ── Create a new project (also creates the 1:1 dataset + draft version + 3 splits) ── */
  .post(
    '/',
    async ({ body, user }) => {
      // Derive modality from task
      const modality = taskToModality(body.task)

      // Create the project
      const [project] = await db
        .insert(projects)
        .values({
          ...body,
          userId: user.id,
        })
        .returning()

      // Create the 1:1 dataset (PK = project.id)
      await db.insert(datasets).values({
        projectId: project.id,
        modality,
      })

      // Create the draft version (versionTag = null)
      const [draftVersion] = await db
        .insert(datasetVersions)
        .values({
          datasetId: project.id, // datasetId references datasets.projectId
        })
        .returning()

      // Create train/validation/test splits for the draft
      await db.insert(datasetSplits).values([
        { datasetVersionId: draftVersion.id, splitType: 'train' },
        { datasetVersionId: draftVersion.id, splitType: 'validation' },
        { datasetVersionId: draftVersion.id, splitType: 'test' },
      ])

      return { project }
    },
    {
      body: t.Object({
        name: t.String(),
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
        name: t.Optional(t.String()),
        description: t.MaybeEmpty(t.String()),
      }),
    },
  )
  /* ── Delete project (cascades to dataset, versions, items, runs, etc.) ── */
  .delete(
    '/:projectId',
    async ({ params, user }) => {
      const deleted = await db
        .delete(projects)
        .where(and(eq(projects.id, params.projectId), eq(projects.userId, user.id)))
        .returning()
      if (deleted.length === 0) return new NotFoundError('Project not found')
      return status(204)
    },
    { auth: true },
  )

import { db } from '@server/db'
import { datasets, datasetVersions, projects, trainingRuns } from '@server/db/schema'
import { cleanupProjectStorage } from '@server/lib/cleanup'
import { ProjectTasks } from '@server/lib/enums'
import { getTaskDescriptor, taskToModality } from '@server/lib/tasks'
import { and, eq } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
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
      const [versionCount, runCount, datasetRow] = await Promise.all([
        db.$count(datasetVersions, eq(datasetVersions.datasetId, project.id)),
        db.$count(trainingRuns, eq(trainingRuns.projectId, project.id)),
        db.query.datasets.findFirst({
          where: { projectId: project.id },
          with: {
            draft: { with: { items: true } },
            versions: { with: { items: true } },
            classes: true,
          },
        }),
      ])

      return { project: { ...project, runCount, versionCount, dataset: datasetRow ?? null } }
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

      // Create the draft version (versionTag = null). Splits are just a
      // column on each item's version-membership row now, so there's
      // nothing else to pre-create here.
      await db.insert(datasetVersions).values({
        datasetId: project.id, // datasetId references datasets.projectId
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
      await db.delete(projects).where(eq(projects.id, project.id))
      return status(204)
    },
    { projectBelongToUser: true },
  )

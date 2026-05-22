import { db } from '@server/db'
import { datasetClasses, datasets, projects } from '@server/db/schema'
import { and, eq } from 'drizzle-orm'
import { Elysia, NotFoundError, t } from 'elysia'
import { betterAuth } from './auth'

export const projectRoutes = new Elysia({ prefix: '/api/projects' })
  .use(betterAuth)
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
          createdAt: true,
        },
      })

      return { projects: rows }
    },
    { auth: true },
  )
  .get(
    '/:projectId',
    async ({ project }) => {
      const datasetCount = await db.$count(datasets, eq(datasets.projectId, project.id))
      const classCount = await db.$count(datasetClasses, eq(datasetClasses.projectId, project.id))

      return { project: { ...project, datasetCount, classCount } }
    },
    {
      projectBelongToUser: true,
    },
  )
  .post(
    '/',
    async ({ body, user }) => {
      const [project] = await db
        .insert(projects)
        .values({
          name: body.name,
          description: body.description,
          userId: user.id,
        })
        .returning()

      return { project }
    },
    {
      body: t.Object({
        name: t.String({ minLength: 1 }),
        description: t.Optional(t.String()),
      }),
      auth: true,
    },
  )
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
        description: t.Optional(t.Nullable(t.String())),
      }),
    },
  )
  .delete(
    '/:projectId',
    async ({ params, user }) => {
      const deleted = await db
        .delete(projects)
        .where(and(eq(projects.id, params.projectId), eq(projects.userId, user.id)))
        .returning()
      if (deleted.length === 0) return new NotFoundError('Project not found')
      return { success: true }
    },
    { auth: true },
  )

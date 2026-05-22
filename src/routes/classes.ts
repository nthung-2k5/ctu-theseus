import { db } from '@server/db'
import { datasetClasses } from '@server/db/schema'
import { eq } from 'drizzle-orm'
import { Elysia, NotFoundError, status, t } from 'elysia'
import { betterAuth } from './auth'

export const classesRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)
  .get(
    '/projects/:projectId/classes',
    async ({ project }) => {
      const rows = await db.query.datasetClasses.findMany({
        where: {
          projectId: project.id,
        },
        orderBy: {
          name: 'asc',
        }
      })

      return { classes: rows }
    },
    { projectBelongToUser: true },
  )
  .post(
    '/projects/:projectId/classes',
    async ({ project, body }) => {
      const existingClass = await db.query.datasetClasses.findFirst({
        where: { name: body.name, projectId: project.id },
      })

      if (existingClass) {
        return status(400, 'Class already exists')
      }

      const [row] = await db
        .insert(datasetClasses)
        .values({
          projectId: project.id,
          name: body.name,
          color: body.color,
        })
        .returning()

      return { class: row }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        name: t.String({ minLength: 1 }),
        color: t.RegExp(/^#([0-9A-F]{3}){1,2}$/i),
      }),
    },
  )
  .patch(
    '/classes/:classId',
    async ({ params, body, user }) => {
      const row = await db.query.datasetClasses.findFirst({
        where: {
          id: params.classId,
        },
        with: {
          projects: {
            where: {
              userId: user.id,
            },
          },
        },
      })

      if (!row) return status(404, 'Class not found')

      const [updated] = await db
        .update(datasetClasses)
        .set(body)
        .where(eq(datasetClasses.id, params.classId))
        .returning()

      return { class: updated }
    },
    {
      auth: true,
      body: t.Object({
        name: t.Optional(t.String({ minLength: 1 })),
        color: t.Optional(t.RegExp(/^#([0-9A-F]{3}){1,2}$/i)),
      }),
    },
  )
  .delete(
    '/classes/:classId',
    async ({ params, user }) => {
      const row = await db.query.datasetClasses.findFirst({
        where: {
          id: params.classId,
        },
        with: {
          projects: {
            where: {
              userId: user.id,
            },
          },
        },
      })

      if (!row) return new NotFoundError('Class not found')

      await db.delete(datasetClasses).where(eq(datasetClasses.id, params.classId))

      return status(204)
    },
    { auth: true },
  )

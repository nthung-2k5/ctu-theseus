import { auth } from '@server/auth'
import { db } from '@server/db'
import { projects } from '@server/db/schema'
import { eq } from 'drizzle-orm'
import Elysia, { t } from 'elysia'

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
  .macro('projectBelongToUser', {
    auth: true,
    params: t.Object({
      projectId: t.String(),
    }),
    async resolve({ status, params, user }) {
      const project = await db.query.projects.findFirst({
        where: eq(projects.id, params.projectId),
      })
      if (!project) return status(404, 'Project not found')
      if (project.userId !== user.id) return status(403, 'Unauthorized')
      return {
        project,
      }
    },
  })

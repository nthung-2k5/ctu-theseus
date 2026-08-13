import { auth } from '@server/auth'
import { db } from '@server/db'
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
      versionId: t.String(),
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
  .macro('runBelongToUser', {
    auth: true,
    params: t.Object({
      runId: t.String(),
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
      itemId: t.String(),
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

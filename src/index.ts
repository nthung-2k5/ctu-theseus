import { staticPlugin } from '@elysia/static'
import { classesRoutes } from '@server/routes/classes'
import { datasetRoutes } from '@server/routes/datasets'
import { projectRoutes } from '@server/routes/projects'
import { Elysia } from 'elysia'
import { auth } from './auth'

const app = new Elysia()
  .onError(({ error }) => {
    console.error(error)
    return 'Internal Server Error'
  })
  .use(
    await staticPlugin({
      prefix: '/',
      bunFullstack: true,
      alwaysStatic: true,
    }),
  )
  .use(
    await staticPlugin({
      prefix: '/data',
      assets: 'data',
      alwaysStatic: true,
    }),
  )
  /* ── Authentication ── */
  .mount(auth.handler)
  /* ── API Routes ── */
  .use(projectRoutes)
  .use(classesRoutes)
  .use(datasetRoutes)
  /* ── SPA Fallback ── */
  // .get('/*', index)
  .listen(3000)

console.log(`CTU Theseus server is running at http://${app.server?.hostname}:${app.server?.port}`)

export type App = typeof app

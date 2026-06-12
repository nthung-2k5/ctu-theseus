import { staticPlugin } from '@elysia/static'
import { classesRoutes } from '@server/routes/classes'
import { datasetRoutes } from '@server/routes/datasets'
import { inferenceRoutes, startInferenceNatsConsumers } from '@server/routes/inference'
import { projectRoutes } from '@server/routes/projects'
import { trainingRoutes } from '@server/routes/training'
import { Elysia } from 'elysia'
import { auth } from './auth'
import { startNatsConsumers } from './lib/microservice'
import { initNats } from './lib/nats'
import { ensureBuckets } from './lib/storage'

// Initialize NATS connection and S3 buckets
await initNats()
await ensureBuckets()
await startNatsConsumers()
await startInferenceNatsConsumers()

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
  /* ── Authentication ── */
  .mount(auth.handler)
  /* ── API Routes ── */
  .use(projectRoutes)
  .use(classesRoutes)
  .use(datasetRoutes)
  .use(trainingRoutes)
  .use(inferenceRoutes)
  /* ── SPA Fallback ── */
  // .get('/*', index)
  .listen({
    port: 3000,
    hostname: '0.0.0.0',
  })

console.log(`CTU Theseus server is running at http://${app.server?.hostname}:${app.server?.port}`)

export type App = typeof app

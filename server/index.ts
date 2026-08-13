import { staticPlugin } from '@elysia/static'
import { telemetry } from '@server/lib/telemetry'
import { classRoutes } from '@server/routes/classes'
import { datasetRoutes } from '@server/routes/datasets'
import { exportRoutes } from '@server/routes/export'
import { inferenceRoutes } from '@server/routes/inference'
import { projectRoutes } from '@server/routes/projects'
import { trainingRoutes } from '@server/routes/training'
import { migrate } from 'drizzle-orm/bun-sql/postgres/migrator'
import { Elysia } from 'elysia'
import { auth } from './auth'
import { db } from './db'
import { startNatsConsumers } from './lib/microservice'
import { closeNats, initNats } from './lib/nats'
import { ensureBuckets } from './lib/storage'

// Initialize NATS connection and S3 buckets
await initNats()
await ensureBuckets()

// Initialize the database and run migrations
await migrate(db, { migrationsFolder: './drizzle' })

// Stops the durable consumer's background fetch loop cleanly on shutdown,
// instead of leaving it running against a closing connection.
const shutdownController = new AbortController()
await startNatsConsumers(shutdownController.signal)

async function shutdown() {
  console.log('[index] Shutting down...')
  shutdownController.abort()
  await closeNats()
  process.exit(0)
}

process.on('SIGTERM', shutdown)
process.on('SIGINT', shutdown)

const app = new Elysia()
  .use(telemetry)
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
  .use(classRoutes)
  .use(datasetRoutes)
  .use(trainingRoutes)
  .use(inferenceRoutes)
  .use(exportRoutes)
  .listen({
    port: 3000,
    hostname: '0.0.0.0',
  })

console.log(`CTU Theseus server is running at http://${app.server?.hostname}:${app.server?.port}`)

export type App = typeof app

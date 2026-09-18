// Aspire TypeScript AppHost
// For more information, see: https://aspire.dev

import { createBuilder } from './.aspire/modules/aspire.mjs'
import CONSTANTS from './schema/constants.json' with { type: 'json' }

const builder = await createBuilder()

const authSecret = process.env.BETTER_AUTH_SECRET
if (!authSecret) {
  throw new Error(
    'BETTER_AUTH_SECRET is not set. Add it to .env (e.g. `openssl rand -base64 32`) before running the AppHost.',
  )
}

const db = await builder
  .addPostgres('database')
  .withDataVolume({
    name: 'ctu-theseus-database',
    isReadOnly: false,
  })
  .withPgAdmin()
  .addDatabase('ctu-theseus-db')

const nats = await builder.addNats('nats').withJetStream().withDataVolume({
  name: 'ctu-theseus-nats',
  isReadOnly: false,
})

const s3AccessKey = await builder.addParameter('s3-access-key', {
  secret: true,
  value: process.env.S3_ACCESS_KEY ?? 'ctu-theseus',
})
const s3SecretKey = await builder.addParameter('s3-secret-key', {
  secret: true,
  value: process.env.S3_SECRET_KEY ?? 'ctu-theseus-secret',
})

const rustfs = await builder
  .addRustFs('rustfs', {
    accessKey: s3AccessKey,
    secretKey: s3SecretKey,
  })
  .withDataVolume({
    name: 'ctu-theseus-rustfs',
    isReadOnly: false,
  })
  .addBuckets([CONSTANTS.BUCKET_DATASETS, CONSTANTS.BUCKET_TRAINING, CONSTANTS.BUCKET_MODELS])

const s3Endpoint = await rustfs.getEndpoint('http') // https://github.com/CommunityToolkit/Aspire/blob/ef0aa306095fb4c7fd0c3ad2fc8c92caa18d5e2d/src/CommunityToolkit.Aspire.Hosting.RustFs/RustFsResource.cs#L12

const AI_WORKER_CUDA_IMAGE = 'nvidia/cuda:13.1.2-runtime-ubuntu24.04'

// Moved to Docker because Ludwig can't work on Windows
const worker = await builder
  .addDockerfileBuilder('ai-worker', './ai_service', async (ctx) => {
    await ctx
      .builder()
      .from(AI_WORKER_CUDA_IMAGE)
      .env('DEBIAN_FRONTEND', 'noninteractive')
      .run(
        'apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*',
      )
      .copyFrom('ghcr.io/astral-sh/uv:latest', '/uv', '/usr/local/bin/uv')
      .copyFrom('ghcr.io/astral-sh/uv:latest', '/uvx', '/usr/local/bin/uvx')
      .env('UV_PROJECT_ENVIRONMENT', '/app/.venv')
      .env('UV_COMPILE_BYTECODE', '1')
      .env('UV_LINK_MODE', 'copy')
      .env('UV_PYTHON', 'python3.12')
      .env('PATH', '/app/.venv/bin:$PATH')
      .workDir('/app')
      .copy('pyproject.toml', 'pyproject.toml')
      .copy('uv.lock', 'uv.lock')
      .copy('.python-version', '.python-version')
      .run('--mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --no-dev')
      .copy('.', '.')
      .run('--mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev')
      .expose(8000)
      .entrypoint(['python', 'main.py'])
  })
  .withContainerRuntimeArgs(['--gpus', 'all'])
  .withHttpEndpoint({
    name: 'health',
    targetPort: 8000,
    env: 'PORT',
  })
  .withHttpHealthCheck({
    path: '/health',
    endpointName: 'health',
  })
  .withReference(nats)
  .withEnvironment('S3_ENDPOINT', s3Endpoint)
  .withEnvironment('S3_ACCESS_KEY', s3AccessKey)
  .withEnvironment('S3_SECRET_KEY', s3SecretKey)
  .waitFor(nats)
  .waitFor(rustfs)
  .withBindMount('./schema', '/schema', { isReadOnly: true })

const gateway = await builder
  .addBunApp('gateway', './server', 'index.ts')
  .withHttpEndpoint({
    port: 3000,
    targetPort: 3000,
    isProxied: false,
    env: 'PORT',
  })
  .withReference(db)
  .withReference(nats)
  .withEnvironment('S3_ENDPOINT', s3Endpoint)
  .withEnvironment('S3_ACCESS_KEY', s3AccessKey)
  .withEnvironment('S3_SECRET_KEY', s3SecretKey)
  .withEnvironment('BETTER_AUTH_SECRET', authSecret)
  .waitFor(db)
  .waitFor(nats)
  .waitFor(rustfs)
  .waitFor(worker)

const web = await builder
  .addViteApp('web', './web')
  .withBun()
  .withEndpoint({
    name: 'http',
    port: 5173,
    isProxied: false,
  })
  .withReference(gateway)
  .waitFor(gateway)
  .withExternalHttpEndpoints()

await builder
  .addYarp('proxy')
  .withConfiguration(async (config) => {
    await config.addCatchAllRoute(web)
    await config.addRoute('/api/{**catch-all}', gateway)
  })
  .waitFor(gateway)
  .waitFor(web)

await builder.build().run()

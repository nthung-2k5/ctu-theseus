// Aspire TypeScript AppHost
// For more information, see: https://aspire.dev

import { createBuilder } from './.aspire/modules/aspire.mjs'
import CONSTANTS from './schema/constants.json' with { type: 'json' }

const builder = await createBuilder()

const jwtSecret = process.env.JWT_SECRET
if (!jwtSecret) {
  throw new Error('JWT_SECRET is not set. Add it to .env (e.g. `openssl rand -base64 32`) before running the AppHost.')
}

const db = await builder
  .addPostgres('database')
  // uuidv7() is a Postgres-side column default (the schema relies on it), so this needs Postgres 18.
  .withImageTag('18')
  .withDataVolume({
    name: 'ctu-theseus-database',
    isReadOnly: false,
  })
  .withPgAdmin()
  .addDatabase('ctu-theseus-db')

const s3AccessKey = await builder.addParameter('s3-access-key', {
  secret: true,
  value: process.env.S3_ACCESS_KEY ?? 'ctu-theseus',
})
const s3SecretKey = await builder.addParameter('s3-secret-key', {
  secret: true,
  value: process.env.S3_SECRET_KEY ?? 'ctu-theseus-secret',
})

// The API runs in a container, so the S3 endpoint Aspire injects into it (S3_ENDPOINT) is a container-network
// hostname. The browser loads images and downloads straight from S3 through presigned URLs, and can only reach
// the host-mapped port. Pin that port so the API can be told the browser-facing address up front
// (S3_PUBLIC_ENDPOINT). Override with S3_HOST_PORT if 9000 is taken on your machine.
const s3HostPort = Number(process.env.S3_HOST_PORT ?? 9000)

const rustfs = await builder
  .addRustFs('rustfs', {
    accessKey: s3AccessKey,
    secretKey: s3SecretKey,
    port: s3HostPort,
  })
  .withDataVolume({
    name: 'ctu-theseus-rustfs',
    isReadOnly: false,
  })
  .addBuckets([CONSTANTS.BUCKET_DATASETS, CONSTANTS.BUCKET_TRAINING, CONSTANTS.BUCKET_MODELS, CONSTANTS.BUCKET_UPLOADS])

const s3Endpoint = await rustfs.getEndpoint('http') // https://github.com/CommunityToolkit/Aspire/blob/ef0aa306095fb4c7fd0c3ad2fc8c92caa18d5e2d/src/CommunityToolkit.Aspire.Hosting.RustFs/RustFsResource.cs#L12

const AI_WORKER_CUDA_IMAGE = 'nvidia/cuda:13.1.2-runtime-ubuntu24.04'

// The whole backend: the REST API, the job queue and Ludwig training/export/inference run in this one
// process (a hard requirement, see theseus.lifespan.assert_single_process). It lives in Docker because
// Ludwig can't work on Windows.
const api = await builder
  .addDockerfileBuilder('api', './ai_service', async (ctx) => {
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
    name: 'http',
    targetPort: 8000,
    env: 'PORT',
  })
  .withHttpHealthCheck({
    path: '/health',
    endpointName: 'http',
  })
  .withReference(db)
  .withEnvironment('S3_ENDPOINT', s3Endpoint)
  .withEnvironment('S3_PUBLIC_ENDPOINT', `http://localhost:${s3HostPort}`)
  .withEnvironment('S3_ACCESS_KEY', s3AccessKey)
  .withEnvironment('S3_SECRET_KEY', s3SecretKey)
  .withEnvironment('JWT_SECRET', jwtSecret)
  // Only needed if the browser reaches the app through a host the proxy doesn't forward (see verify_origin).
  .withEnvironment('ALLOWED_ORIGINS', process.env.ALLOWED_ORIGINS ?? '')
  .waitFor(db)
  .waitFor(rustfs)
  .withBindMount('./schema', '/schema', { isReadOnly: true })

const web = await builder
  .addViteApp('web', './web')
  .withBun()
  .withEndpoint({
    name: 'http',
    port: 5173,
    isProxied: false,
  })
  .waitFor(api)
  .withExternalHttpEndpoints()

await builder
  .addYarp('proxy')
  .withConfiguration(async (config) => {
    await config.addCatchAllRoute(web)
    // A plain container has no service discovery, so route to its endpoint rather than the resource.
    await config.addRoute('/api/{**catch-all}', await api.getEndpoint('http'))
  })
  .waitFor(api)
  .waitFor(web)

await builder.build().run()

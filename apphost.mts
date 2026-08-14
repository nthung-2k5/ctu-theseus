// Aspire TypeScript AppHost
// For more information, see: https://aspire.dev

import { createBuilder } from './.aspire/modules/aspire.mjs'
import CONSTANTS from './schema/constants.json' with { type: 'json' }

const builder = await createBuilder()

const authSecret = process.env.BETTER_AUTH_SECRET ?? 'fJp314Y1mVsTWzz2VJkLj2QcZ8wrAwgB'

const db = await builder
  .addPostgres('database')
  .withDataVolume({
    name: 'ctu-theseus-database',
    isReadOnly: false,
  })
  .addDatabase('ctu-theseus-db')

const nats = await builder.addNats('nats').withJetStream().withDataVolume({
  name: 'ctu-theseus-nats',
  isReadOnly: false,
})

const s3AccessKey = await builder.addParameter('s3-access-key', { secret: true, value: 'ctu-theseus' })
const s3SecretKey = await builder.addParameter('s3-secret-key', { secret: true, value: 'ctu-theseus-secret' })

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

const worker = await builder
  .addPythonModule('ai-worker', './ai_service', 'main')
  .withUv()
  .withHttpEndpoint({
    name: 'health',
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

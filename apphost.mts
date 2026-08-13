// Aspire TypeScript AppHost
// For more information, see: https://aspire.dev

import { createBuilder } from './.aspire/modules/aspire.mjs'
import CONSTANTS from './schema/constants.json'

const builder = await createBuilder()

const jwtSecret = process.env.JWT_SECRET ?? '+hSRqW0ZVKDuzB7/K11Rh+frp5GMZrIE73K2hKFmjj4='
const authSecret = process.env.BETTER_AUTH_SECRET ?? 'fJp314Y1mVsTWzz2VJkLj2QcZ8wrAwgB'

const db = await builder
  .addPostgres('database')
  .withDataVolume({
    name: 'ctu-theseus-database',
    isReadOnly: false,
  })
  .addDatabase('ctu-theseus')

const nats = await builder.addNats('nats').withJetStream().withDataVolume({
  name: 'ctu-theseus-nats',
  isReadOnly: false,
})

var s3AccessKey = await builder.addParameter('S3_ACCESS_KEY', { secret: true, value: 'ctu-theseus' })
var s3SecretKey = await builder.addParameter('S3_SECRET_KEY', { secret: true, value: 'ctu-theseus-secret' })

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

var s3Endpoint = await rustfs.getEndpoint('http') // https://github.com/CommunityToolkit/Aspire/blob/ef0aa306095fb4c7fd0c3ad2fc8c92caa18d5e2d/src/CommunityToolkit.Aspire.Hosting.RustFs/RustFsResource.cs#L12

var worker = await builder
  .addPythonModule('ai-worker', './ai_service', 'ai_service.main')
  .withUv()
  .withHttpEndpoint({
    port: 8000,
    targetPort: 8000,
    name: 'health',
    isProxied: false,
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

var gateway = await builder
  .addBunApp('gateway', './src', 'index.ts')
  .withHttpEndpoint({
    port: 3000,
    targetPort: 3000,
    isProxied: false,
  })
  .withReference(db)
  .withReference(nats)
  .withEnvironment('S3_ENDPOINT', s3Endpoint)
  .withEnvironment('S3_ACCESS_KEY', s3AccessKey)
  .withEnvironment('S3_SECRET_KEY', s3SecretKey)
  .withEnvironment('JWT_SECRET', jwtSecret)
  .withEnvironment('BETTER_AUTH_SECRET', authSecret)
  .waitFor(db)
  .waitFor(nats)
  .waitFor(rustfs)
  .waitFor(worker)

await builder
  .addViteApp('web', './public')
  .withBun()
  .withEndpoint({
    name: 'http',
    port: 5173,
    isProxied: false,
  })
  .withReference(gateway)
  .waitFor(gateway)
  .withExternalHttpEndpoints()

await builder.build().run()

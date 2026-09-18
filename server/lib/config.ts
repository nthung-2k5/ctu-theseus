function env(...names: string[]): string | undefined {
  for (const name of names) {
    const value = process.env[name]
    if (value) return value
  }
  return undefined
}

const isProduction = process.env.NODE_ENV === 'production'

/**
 * Dev-only fallback. In production a missing value is a misconfiguration, not
 * something to paper over: silently defaulting `databaseUrl` points a prod
 * gateway at a local Postgres, and defaulting the S3 keys surfaces as an
 * opaque 403 deep inside a training run instead of at boot.
 */
function required(name: string, value: string | undefined, devFallback: string): string {
  if (value) return value
  if (isProduction) throw new Error(`Missing required environment variable: ${name}`)
  return devFallback
}

export const config = {
  databaseUrl: required(
    'CTU_THESEUS_DB_URI',
    env('CTU_THESEUS_DB_URI'),
    'postgres://theseus:theseus@localhost:5432/theseus',
  ),
  natsUri: required('NATS_URI', env('NATS_URI'), 'nats://localhost:4222'),
  s3Endpoint: required('S3_ENDPOINT', env('S3_ENDPOINT'), 'http://localhost:9000'),
  // Keep these two in sync with ai_service/config.py — the worker and the
  // gateway must agree, or the worker authenticates with credentials nothing
  // issued the moment either runs outside Aspire.
  s3AccessKey: required('S3_ACCESS_KEY', env('S3_ACCESS_KEY'), 'ctu-theseus'),
  s3SecretKey: required('S3_SECRET_KEY', env('S3_SECRET_KEY'), 'ctu-theseus-secret'),
}

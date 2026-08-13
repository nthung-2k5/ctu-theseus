function env(...names: string[]): string | undefined {
  for (const name of names) {
    const value = process.env[name]
    if (value) return value
  }
  return undefined
}

export const config = {
  databaseUrl: env('DATABASE_URI') ?? 'postgres://theseus:theseus@localhost:5432/theseus',
  natsUri: env('NATS_URI') ?? 'nats://localhost:4222',
  s3Endpoint: env('S3_ENDPOINT') ?? 'http://localhost:9000',
  s3AccessKey: env('S3_ACCESS_KEY') ?? 'ctu-theseus',
  s3SecretKey: env('S3_SECRET_KEY') ?? 'ctu-theseus-secret',
}

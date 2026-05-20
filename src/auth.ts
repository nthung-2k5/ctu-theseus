import { drizzleAdapter } from '@better-auth/drizzle-adapter'
import { db } from '@server/db'
import { betterAuth } from 'better-auth'
import { admin } from 'better-auth/plugins'

export const auth = betterAuth({
  database: drizzleAdapter(db, {
    provider: 'pg',
    usePlural: true,
  }),
  basePath: '/api/auth',
  advanced: {
    database: { generateId: 'uuid' },
  },
  emailAndPassword: {
    enabled: true,
  },
  experimental: { joins: true },
  plugins: [admin()],
})

import { db } from '@server/db'
import { betterAuth } from 'better-auth'
import { admin } from 'better-auth/plugins'
import { drizzleAdapter } from 'better-auth-drizzle-adapter'
import * as schema from './db/schema'

export const auth = betterAuth({
  database: drizzleAdapter(db, {
    provider: 'pg',
    usePlural: true,
    schema, // Wait until Better Auth merges PR#9489 (https://github.com/better-auth/better-auth/pull/9489)
  }),
  baseURL: {
    allowedHosts: ['*.dev.localhost:*', 'localhost:*'],
    protocol: 'auto',
    fallback: 'http://localhost:3000',
  },
  basePath: '/api/auth',
  advanced: {
    database: { generateId: 'uuid' },
    // Force Better Auth to trust the headers passed by Aspire's proxy
    trustedProxyHeaders: true,
  },
  emailAndPassword: {
    enabled: true,
  },
  experimental: { joins: true },
  plugins: [admin()],
})

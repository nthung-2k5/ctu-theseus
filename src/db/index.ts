import { config } from '@server/lib/config'
import { SQL } from 'bun'
import { drizzle } from 'drizzle-orm/bun-sql'
import relations from './relations'

/**
 * Raw postgres client for migrations and raw queries.
 */
export const sql = new SQL(config.databaseUrl)

/**
 * Drizzle ORM instance with schema inference.
 */
export const db = drizzle({ client: sql, relations })

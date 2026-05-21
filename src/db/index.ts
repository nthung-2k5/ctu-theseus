import { SQL } from 'bun'
import { drizzle } from 'drizzle-orm/bun-sql'
import { relations } from './relations'

const DATABASE_URL = process.env.DATABASE_URL ?? 'postgres://theseus:theseus@localhost:5432/theseus'

/**
 * Raw postgres client for migrations and raw queries.
 */
export const sql = new SQL(DATABASE_URL)

/**
 * Drizzle ORM instance with schema inference.
 */
export const db = drizzle({ client: sql, relations })

import { treaty } from '@elysia/eden'
import type { App } from '@server'
import { createEdenOptionsProxy, createEdenTanStackQuery } from 'eden-tanstack-react-query'

// The `elysia` package resolves to two physically distinct copies here
// (server/node_modules vs web/node_modules — there is no Bun workspace).
// `Elysia` has private/protected members, so TS compares the two nominally
// and treats them as incompatible even though they're the same version.
export const { EdenProvider, useEden, useEdenClient } = createEdenTanStackQuery<App>()

export const client = treaty<App>(window.location.origin, {
  fetch: {
    credentials: 'include',
  },
})

export const rest = client.api

export const edenOptions = createEdenOptionsProxy<App>({ client })

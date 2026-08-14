import { queryOptions } from '@tanstack/react-query'
import { createAuthClient } from 'better-auth/react'

export const authClient = createAuthClient()

export type Session = Awaited<ReturnType<typeof authClient.getSession>>['data']

/**
 * Routes the session through TanStack Query so router `beforeLoad` guards
 * (which run outside the React tree and can't call the `useSession` hook)
 * and components share one cache. `AppShell` still uses `useSession()`
 * directly for the reactive avatar/name display.
 */
export const sessionQueryOptions = queryOptions({
  queryKey: ['session'],
  queryFn: async (): Promise<Session> => (await authClient.getSession()).data,
  staleTime: 5 * 60_000,
  retry: false,
})

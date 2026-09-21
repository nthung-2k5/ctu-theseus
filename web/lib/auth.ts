import { queryOptions } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import { getMe } from './api/generated/auth/auth'
import type { UserOut } from './api/generated/models'

export type SessionUser = UserOut
export type Session = SessionUser | null

/**
 * Routes the session through TanStack Query so router `beforeLoad` guards (which run outside the React tree)
 * and components share one cache. `null` means signed out: a 401 here has already been through the axios
 * refresh interceptor, so it is a real "no session", not an expired access token.
 */
export const sessionQueryOptions = queryOptions({
  queryKey: ['session'],
  queryFn: async (): Promise<Session> => {
    try {
      return (await getMe()).user
    } catch (error) {
      if (isAxiosError(error) && error.response?.status === 401) return null
      throw error
    }
  },
  staleTime: 5 * 60_000,
  retry: false,
})

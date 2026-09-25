import { sessionQueryOptions } from '@public/lib/auth'
import { safeRedirect } from '@public/lib/safeRedirect'
import { AuthPage } from '@public/pages/AuthPage'
import { Landing } from '@public/pages/landing/Landing'
import { createRoute, redirect } from '@tanstack/react-router'
import { rootRoute } from './__root'

/** Public introduction page. Signed-in users skip straight to their projects. */
export const landingRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.ensureQueryData(sessionQueryOptions)
    if (session) throw redirect({ to: '/projects' })
  },
  component: Landing,
})

/** Wraps guest-only routes (login/register). Already-authenticated users are bounced to their projects. */
export const guestRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: '_guest',
  beforeLoad: async ({ context, location }) => {
    const session = await context.queryClient.ensureQueryData(sessionQueryOptions)
    if (session) {
      const target = new URLSearchParams(location.searchStr).get('redirect') ?? undefined
      throw redirect({ href: safeRedirect(target) })
    }
  },
})

export const loginRoute = createRoute({
  getParentRoute: () => guestRoute,
  path: '/login',
  validateSearch: (search: Record<string, unknown>): { redirect?: string; mode?: 'signin' | 'register' } => ({
    redirect: typeof search.redirect === 'string' ? search.redirect : undefined,
    mode: search.mode === 'register' ? 'register' : undefined,
  }),
  component: AuthPage,
})

/** Old URL: register is now a mode of the login page. */
export const registerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/register',
  beforeLoad: () => {
    throw redirect({ to: '/login', search: { mode: 'register' } })
  },
})

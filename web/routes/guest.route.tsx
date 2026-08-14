import { sessionQueryOptions } from '@public/lib/auth'
import { LoginPage } from '@public/pages/LoginPage'
import { RegisterPage } from '@public/pages/RegisterPage'
import { createRoute, redirect } from '@tanstack/react-router'
import { rootRoute } from './__root'

/** Wraps guest-only routes (login, register). Already-authenticated users are bounced to the dashboard. */
export const guestRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: '_guest',
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.ensureQueryData(sessionQueryOptions)
    if (session) throw redirect({ to: '/' })
  },
})

export const loginRoute = createRoute({
  getParentRoute: () => guestRoute,
  path: '/login',
  validateSearch: (search: Record<string, unknown>): { redirect?: string } => ({
    redirect: typeof search.redirect === 'string' ? search.redirect : undefined,
  }),
  component: LoginPage,
})

export const registerRoute = createRoute({
  getParentRoute: () => guestRoute,
  path: '/register',
  component: RegisterPage,
})

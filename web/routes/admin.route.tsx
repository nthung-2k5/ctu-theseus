import { AdminLayout } from '@public/layouts/AdminLayout'
import { sessionQueryOptions } from '@public/lib/auth'
import { UsersPage } from '@public/pages/admin/UsersPage'
import { createRoute, redirect } from '@tanstack/react-router'
import { appRoute } from './app.route'

/**
 * Everything under /admin. Non-admins are bounced to the dashboard here for UX only: the real
 * enforcement is the API, where every /api/admin route is behind `require_admin`.
 */
export const adminRoute = createRoute({
  getParentRoute: () => appRoute,
  path: 'admin',
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.ensureQueryData(sessionQueryOptions)
    if (session?.role !== 'admin') throw redirect({ to: '/projects' })
  },
  component: AdminLayout,
})

export const adminIndexRoute = createRoute({
  getParentRoute: () => adminRoute,
  path: '/',
  beforeLoad: () => {
    throw redirect({ to: '/admin/users' })
  },
})

export const adminUsersRoute = createRoute({
  getParentRoute: () => adminRoute,
  path: '/users',
  component: UsersPage,
})

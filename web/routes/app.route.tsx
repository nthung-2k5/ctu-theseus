import { AppShell } from '@public/layouts/AppShell'
import { isAdmin, sessionQueryOptions } from '@public/lib/auth'
import { DashboardPage } from '@public/pages/DashboardPage'
import { ImportedModelsPage } from '@public/pages/ImportedModelsPage'
import { SettingsPage } from '@public/pages/SettingsPage'
import { createRoute, Outlet, redirect } from '@tanstack/react-router'
import { rootRoute } from './__root'

/**
 * Wraps every authenticated route in the app shell. Unauthenticated users are sent to /login with a redirect-back
 * target. Admins are kept in /admin: the rest of the app (projects, training, models, settings) is for users. This
 * is a UX separation; the API itself is not restricted by role outside /api/admin.
 */
export const appRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: '_app',
  beforeLoad: async ({ context, location }) => {
    const session = await context.queryClient.ensureQueryData(sessionQueryOptions)
    if (!session) throw redirect({ to: '/login', search: { redirect: location.href } })
    const inAdminArea = location.pathname === '/admin' || location.pathname.startsWith('/admin/')
    if (isAdmin(session) && !inAdminArea) throw redirect({ to: '/admin' })
  },
  component: () => (
    <AppShell>
      <Outlet />
    </AppShell>
  ),
})

export const dashboardRoute = createRoute({
  getParentRoute: () => appRoute,
  path: '/projects',
  component: DashboardPage,
})

/** The models the user imported themselves. */
export const importedModelsRoute = createRoute({
  getParentRoute: () => appRoute,
  path: '/imported-models',
  component: ImportedModelsPage,
})

export const settingsRoute = createRoute({
  getParentRoute: () => appRoute,
  path: '/settings',
  component: SettingsPage,
})

/** Old URL: API keys are a section of Settings now. */
export const apiKeysRoute = createRoute({
  getParentRoute: () => appRoute,
  path: '/settings/api-keys',
  component: SettingsPage,
})

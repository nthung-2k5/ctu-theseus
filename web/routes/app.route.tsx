import { AppShell } from '@public/layouts/AppShell'
import { sessionQueryOptions } from '@public/lib/auth'
import { DashboardPage } from '@public/pages/DashboardPage'
import { SettingsPage } from '@public/pages/SettingsPage'
import { createRoute, Outlet, redirect } from '@tanstack/react-router'
import { rootRoute } from './__root'

/** Wraps every authenticated route in the app shell. Unauthenticated users are sent to /login with a redirect-back target. */
export const appRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: '_app',
  beforeLoad: async ({ context, location }) => {
    const session = await context.queryClient.ensureQueryData(sessionQueryOptions)
    if (!session) throw redirect({ to: '/login', search: { redirect: location.href } })
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

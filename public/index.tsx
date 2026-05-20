import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { createRoot } from 'react-dom/client'
import { Route, Router, Switch } from 'wouter'
import { useHashLocation } from 'wouter/use-hash-location'

import '@public/global.css'
import { AuthGuard } from '@public/components/AuthGuard'
import { GuestGuard } from '@public/components/GuestGuard'
import { AppShell } from '@public/layouts/AppShell'
import { DashboardPage } from '@public/pages/DashboardPage'
import { LoginPage } from '@public/pages/LoginPage'
import { RegisterPage } from '@public/pages/RegisterPage'
import { theme } from '@public/theme'

function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Notifications position="top-right" />
      <Router hook={useHashLocation}>
        <Switch>
          {/* ── Guest-only routes ── */}
          <Route path="/login">
            <GuestGuard>
              <LoginPage />
            </GuestGuard>
          </Route>
          <Route path="/register">
            <GuestGuard>
              <RegisterPage />
            </GuestGuard>
          </Route>

          {/* ── Protected routes ── */}
          <Route path="/" nest>
            <AuthGuard>
              <AppShell>
                <Switch>
                  <Route path="/" component={DashboardPage} />
                </Switch>
              </AppShell>
            </AuthGuard>
          </Route>
        </Switch>
      </Router>
    </MantineProvider>
  )
}

const root = createRoot(document.getElementById('root') as HTMLElement)
root.render(<App />)

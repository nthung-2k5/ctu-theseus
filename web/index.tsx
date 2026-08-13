import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { createRoot } from 'react-dom/client'
import { Route, Router, Switch } from 'wouter'

import '@public/global.css'
import { ModalsProvider } from '@mantine/modals'
import { AuthGuard } from '@public/components/AuthGuard'
import { GuestGuard } from '@public/components/GuestGuard'
import { ProjectProvider } from '@public/components/ProjectContext'
import { AppShell } from '@public/layouts/AppShell'
import { ClassesPage } from '@public/pages/ClassesPage'
import { DashboardPage } from '@public/pages/DashboardPage'
import { DataPage } from '@public/pages/DataPage'
import { DatasetPage } from '@public/pages/DatasetPage'
import { InferencePage } from '@public/pages/InferencePage'
import { LoginPage } from '@public/pages/LoginPage'
import { ModelsPage } from '@public/pages/ModelsPage'
import { ProjectPage } from '@public/pages/ProjectPage'
import { RegisterPage } from '@public/pages/RegisterPage'
import { TrainingPage } from '@public/pages/TrainingPage'
import { theme } from '@public/theme'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MantineProvider theme={theme} defaultColorScheme="dark">
        <ModalsProvider>
          <Notifications position="top-right" />
          <Router>
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
                      <Route path="/project/:id" nest>
                        <ProjectProvider>
                          <Route path="/" component={ProjectPage} />
                          <Route path="/data" component={DataPage} />
                          <Route path="/dataset" component={DatasetPage} />
                          <Route path="/classes" component={ClassesPage} />
                          <Route path="/training" component={TrainingPage} />
                          <Route path="/models" component={ModelsPage} />
                          <Route path="/inference" component={InferencePage} />
                        </ProjectProvider>
                      </Route>
                    </Switch>
                  </AppShell>
                </AuthGuard>
              </Route>
            </Switch>
          </Router>
        </ModalsProvider>
      </MantineProvider>
    </QueryClientProvider>
  )
}

const root = createRoot(document.getElementById('root') as HTMLElement)
root.render(<App />)

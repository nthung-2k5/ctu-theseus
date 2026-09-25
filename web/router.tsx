import { Button, Center, Loader, Stack, Text, Title } from '@mantine/core'
import type { QueryClient } from '@tanstack/react-query'
import { createRouter, Link } from '@tanstack/react-router'
import { routeTree } from './routes'

function NotFound() {
  return (
    <Center mih="100vh">
      <Stack align="center" gap="sm">
        <Title order={1}>404</Title>
        <Text c="dimmed">This page doesn't exist.</Text>
        <Button component={Link} to="/projects" mt="sm">
          Back to projects
        </Button>
      </Stack>
    </Center>
  )
}

function ErrorFallback({ error }: { error: Error }) {
  return (
    <Center mih="100vh">
      <Stack align="center" gap="sm" maw={480}>
        <Title order={2}>Something went wrong</Title>
        <Text c="dimmed" ta="center">
          {error.message || 'An unexpected error occurred.'}
        </Text>
        <Button component={Link} to="/projects" mt="sm">
          Back to projects
        </Button>
      </Stack>
    </Center>
  )
}

export function createAppRouter(queryClient: QueryClient) {
  return createRouter({
    routeTree,
    context: { queryClient },
    defaultPreload: 'intent',
    defaultPendingComponent: () => (
      <Center mih="100vh">
        <Stack align="center" gap="sm">
          <Loader size="lg" color="primary" />
          <Text size="sm" c="dimmed">
            Loading...
          </Text>
        </Stack>
      </Center>
    ),
    defaultNotFoundComponent: NotFound,
    defaultErrorComponent: ({ error }) => (
      <ErrorFallback error={error instanceof Error ? error : new Error(String(error))} />
    ),
  })
}

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof createAppRouter>
  }
}

import { Center, Loader, Stack, Text } from '@mantine/core'
import { authClient } from '@public/lib/auth'
import type { ReactNode } from 'react'
import { useEffect } from 'react'
import { useLocation } from 'wouter'

interface AuthGuardProps {
  children: ReactNode
}

/**
 * Wraps protected routes. On mount it checks the session.
 * While loading it shows a spinner; if not authenticated it redirects to /login.
 */
export function AuthGuard({ children }: AuthGuardProps) {
  const [, setLocation] = useLocation()
  const { data: session, isPending } = authClient.useSession()

  useEffect(() => {
    if (!isPending && !session) {
      setLocation('/login')
    }
  }, [session, isPending, setLocation])

  if (isPending) {
    return (
      <Center mih="100vh">
        <Stack align="center" gap="sm">
          <Loader size="lg" color="primary" />
          <Text size="sm" c="dimmed">
            Checking session...
          </Text>
        </Stack>
      </Center>
    )
  }

  if (!session) {
    return null
  }

  return <>{children}</>
}

import { Center, Loader, Stack, Text } from '@mantine/core'
import { authClient } from '@public/lib/auth'
import type { ReactNode } from 'react'
import { useEffect } from 'react'
import { useLocation } from 'wouter'

interface GuestGuardProps {
  children: ReactNode
}

/**
 * Wraps guest-only routes (login, register).
 * If the user is already authenticated, redirects to the dashboard.
 */
export function GuestGuard({ children }: GuestGuardProps) {
  const [, setLocation] = useLocation()

  const { data: session, isPending } = authClient.useSession()

  useEffect(() => {
    if (!isPending && session) {
      setLocation('/')
    }
  }, [isPending, session, setLocation])

  if (isPending) {
    return (
      <Center mih="100vh">
        <Stack align="center" gap="sm">
          <Loader size="lg" />
          <Text>Loading session...</Text>
        </Stack>
      </Center>
    )
  }

  return <>{children}</>
}

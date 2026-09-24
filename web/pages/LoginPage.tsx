import { Anchor, Box, Button, Card, Center, PasswordInput, Stack, Text, TextInput, Title } from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { EnvelopeSimpleIcon, LockIcon } from '@phosphor-icons/react'
import { apiErrorMessage } from '@public/lib/api/client'
import { login } from '@public/lib/api/generated/auth/auth'
import { type SessionUser, sessionQueryOptions } from '@public/lib/auth'
import { useQueryClient } from '@tanstack/react-query'
import { getRouteApi, useNavigate } from '@tanstack/react-router'

const routeApi = getRouteApi('/_guest/login')

export function LoginPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { redirect } = routeApi.useSearch()

  const form = useForm({
    initialValues: { email: '', password: '' },
    validate: {
      email: (v) => (/^\S+@\S+$/.test(v) ? null : 'Invalid email address'),
      password: (v) => (v.length >= 6 ? null : 'Password must be at least 6 characters'),
    },
  })

  const handleSubmit = async (values: typeof form.values) => {
    let user: SessionUser
    try {
      user = (await login(values)).user
    } catch (error) {
      notifications.show({ title: 'Login failed', message: apiErrorMessage(error, 'Login failed'), color: 'red' })
      return
    }
    notifications.show({ title: 'Welcome back', message: `Signed in as ${user.name}`, color: 'green' })
    // Seed the cache: the route guards read it via ensureQueryData, which would otherwise return the stale signed-out `null`.
    queryClient.setQueryData(sessionQueryOptions.queryKey, user)
    navigate({ to: redirect ?? '/' })
  }

  return (
    <Center mih="100vh" bg="dark.8">
      <Card shadow="xl" padding="xl" radius="lg" w={420} withBorder>
        <Stack gap="lg">
          <Box ta="center">
            <Title order={2} c="primary">
              CTU Theseus
            </Title>
            <Text size="sm" c="dimmed" mt={4}>
              Sign in to your account
            </Text>
          </Box>

          <form onSubmit={form.onSubmit(handleSubmit)}>
            <Stack gap="md">
              <TextInput
                label="Email"
                placeholder="you@example.com"
                leftSection={<EnvelopeSimpleIcon size={18} />}
                {...form.getInputProps('email')}
              />
              <PasswordInput
                label="Password"
                placeholder="Your password"
                leftSection={<LockIcon size={18} />}
                {...form.getInputProps('password')}
              />
              <Button type="submit" fullWidth>
                Sign in
              </Button>
            </Stack>
          </form>

          <Text ta="center" size="sm" c="dimmed">
            Don't have an account?{' '}
            <Anchor component="button" type="button" size="sm" onClick={() => navigate({ to: '/register' })}>
              Create one
            </Anchor>
          </Text>
        </Stack>
      </Card>
    </Center>
  )
}

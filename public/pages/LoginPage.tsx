import { Anchor, Box, Button, Card, Center, PasswordInput, Stack, Text, TextInput, Title } from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { EnvelopeSimpleIcon, LockIcon } from '@phosphor-icons/react'
import { authClient } from '@public/lib/auth'
import { useLocation } from 'wouter'

export function LoginPage() {
  const [, setLocation] = useLocation()

  const form = useForm({
    initialValues: { email: '', password: '' },
    validate: {
      email: (v) => (/^\S+@\S+$/.test(v) ? null : 'Invalid email address'),
      password: (v) => (v.length >= 6 ? null : 'Password must be at least 6 characters'),
    },
  })

  const handleSubmit = async (values: typeof form.values) => {
    const { data, error } = await authClient.signIn.email(values)
    if (error) {
      notifications.show({ title: 'Login failed', message: error.message ?? 'Login failed', color: 'red' })
      return
    }
    notifications.show({ title: 'Welcome back', message: `Signed in as ${data.user.name}`, color: 'green' })
    setLocation('/')
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
            <Anchor component="button" type="button" size="sm" onClick={() => setLocation('/register')}>
              Create one
            </Anchor>
          </Text>
        </Stack>
      </Card>
    </Center>
  )
}

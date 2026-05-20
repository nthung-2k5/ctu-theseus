import { Anchor, Box, Button, Card, Center, PasswordInput, Stack, Text, TextInput, Title } from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { EnvelopeSimpleIcon, LockIcon, UserCircleIcon } from '@phosphor-icons/react'
import { authClient } from '@public/lib/auth'
import { useLocation } from 'wouter'

export function RegisterPage() {
  const [, setLocation] = useLocation()

  const form = useForm({
    initialValues: { name: '', email: '', password: '', confirmPassword: '' },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Name is required'),
      email: (v) => (/^\S+@\S+$/.test(v) ? null : 'Invalid email address'),
      password: (v) => (v.length >= 6 ? null : 'Password must be at least 6 characters'),
      confirmPassword: (v, values) => (v === values.password ? null : 'Passwords do not match'),
    },
  })

  const handleSubmit = async (values: typeof form.values) => {
    const { data, error } = await authClient.signUp.email({
      email: values.email,
      password: values.password,
      name: values.name,
    })
    if (error) {
      notifications.show({
        title: 'Registration failed',
        message: error.message ?? 'Registration failed',
        color: 'red',
      })
      return
    }
    notifications.show({ title: 'Account created', message: `Welcome, ${data.user.name}`, color: 'green' })
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
              Create a new account
            </Text>
          </Box>

          <form onSubmit={form.onSubmit(handleSubmit)}>
            <Stack gap="md">
              <TextInput
                label="Full name"
                placeholder="Your name"
                leftSection={<UserCircleIcon size={18} />}
                {...form.getInputProps('name')}
              />
              <TextInput
                label="Email"
                placeholder="you@example.com"
                leftSection={<EnvelopeSimpleIcon size={18} />}
                {...form.getInputProps('email')}
              />
              <PasswordInput
                label="Password"
                placeholder="Create a password"
                leftSection={<LockIcon size={18} />}
                {...form.getInputProps('password')}
              />
              <PasswordInput
                label="Confirm password"
                placeholder="Repeat your password"
                leftSection={<LockIcon size={18} />}
                {...form.getInputProps('confirmPassword')}
              />
              <Button type="submit" fullWidth>
                Create account
              </Button>
            </Stack>
          </form>

          <Text ta="center" size="sm" c="dimmed">
            Already have an account?{' '}
            <Anchor component="button" type="button" size="sm" onClick={() => setLocation('/login')}>
              Sign in
            </Anchor>
          </Text>
        </Stack>
      </Card>
    </Center>
  )
}

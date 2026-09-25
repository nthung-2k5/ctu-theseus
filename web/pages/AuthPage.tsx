import {
  Alert,
  Anchor,
  Button,
  Paper,
  PasswordInput,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { ArrowLeftIcon, LockIcon } from '@phosphor-icons/react'
import { apiErrorMessage } from '@public/lib/api/client'
import { login, register } from '@public/lib/api/generated/auth/auth'
import { type SessionUser, sessionQueryOptions } from '@public/lib/auth'
import { safeRedirect } from '@public/lib/safeRedirect'
import { NeuralCanvas } from '@public/pages/landing/NeuralCanvas'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { getRouteApi, Link, useRouter } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_guest/login')

interface AuthValues {
  name: string
  email: string
  password: string
  confirmPassword: string
}

/** Sign in / create account on one glass card over the animated network. Mode lives in the URL (?mode=). */
export function AuthPage() {
  const { redirect: target, mode = 'signin' } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const router = useRouter()
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const isRegister = mode === 'register'

  const form = useForm<AuthValues>({
    initialValues: { name: '', email: '', password: '', confirmPassword: '' },
    validate: {
      name: (v) => (isRegister && v.trim().length === 0 ? 'Name is required' : null),
      email: (v) => (/^\S+@\S+$/.test(v) ? null : 'Invalid email address'),
      password: (v) => (v.length >= 6 ? null : 'Password must be at least 6 characters'),
      confirmPassword: (v, values) => (isRegister && v !== values.password ? 'Passwords do not match' : null),
    },
  })

  const auth = useMutation({
    mutationFn: async (values: AuthValues): Promise<SessionUser> =>
      isRegister
        ? (await register({ email: values.email, password: values.password, name: values.name })).user
        : (await login({ email: values.email, password: values.password })).user,
    onSuccess: (user) => {
      notifications.show({
        title: isRegister ? 'Account created' : 'Welcome back',
        message: isRegister ? `Welcome, ${user.name}` : `Signed in as ${user.name}`,
        color: 'green',
      })
      // Seed the cache: route guards read it via ensureQueryData, which would otherwise return the stale signed-out `null`.
      queryClient.setQueryData(sessionQueryOptions.queryKey, user)
      router.history.push(safeRedirect(target))
    },
    onError: (e) => setError(apiErrorMessage(e, isRegister ? 'Registration failed' : 'Login failed')),
  })

  const switchMode = (next: string) => {
    setError(null)
    void navigate({ search: (prev) => ({ ...prev, mode: next as 'signin' | 'register' }), replace: true })
  }

  return (
    <div
      style={{
        position: 'relative',
        minHeight: '100vh',
        display: 'grid',
        placeItems: 'center',
        overflow: 'hidden',
        background: 'radial-gradient(1200px 600px at 20% 0%, #14305e55, transparent), #070b14',
      }}
    >
      <NeuralCanvas density={0.55} interactive={false} />
      <Anchor
        component={Link}
        to="/"
        size="sm"
        c="dimmed"
        style={{ position: 'absolute', top: 20, left: 24, zIndex: 2, display: 'flex', gap: 6, alignItems: 'center' }}
      >
        <ArrowLeftIcon size={14} /> CTU Theseus
      </Anchor>
      <Paper
        p="xl"
        w={{ base: '92%', sm: 400 }}
        style={{
          position: 'relative',
          zIndex: 2,
          backdropFilter: 'blur(14px)',
          background: 'rgba(15,23,42,.72)',
          borderColor: 'rgba(148,163,184,.18)',
        }}
      >
        <form
          onSubmit={form.onSubmit((values) => {
            setError(null)
            auth.mutate(values)
          })}
          noValidate
        >
          <Stack gap="md">
            <div>
              <Title order={3}>{isRegister ? 'Create your workspace' : 'Welcome back'}</Title>
              <Text size="sm" c="dimmed">
                {isRegister ? 'Start curating datasets and training models.' : 'Sign in to your projects.'}
              </Text>
            </div>
            <SegmentedControl
              fullWidth
              size="xs"
              value={mode}
              onChange={switchMode}
              data={[
                { value: 'signin', label: 'Sign in' },
                { value: 'register', label: 'Create account' },
              ]}
            />
            {error && (
              <Alert color="red" p="xs" role="alert">
                {error}
              </Alert>
            )}
            {isRegister && (
              <TextInput label="Full name" autoComplete="name" data-autofocus {...form.getInputProps('name')} />
            )}
            <TextInput
              label="Email"
              placeholder="you@example.com"
              autoComplete="email"
              data-autofocus={!isRegister || undefined}
              {...form.getInputProps('email')}
            />
            <PasswordInput
              label="Password"
              autoComplete={isRegister ? 'new-password' : 'current-password'}
              {...form.getInputProps('password')}
            />
            {isRegister && (
              <PasswordInput
                label="Confirm password"
                autoComplete="new-password"
                {...form.getInputProps('confirmPassword')}
              />
            )}
            <Button type="submit" loading={auth.isPending} leftSection={<LockIcon size={15} />}>
              {isRegister ? 'Create account' : 'Sign in'}
            </Button>
          </Stack>
        </form>
      </Paper>
    </div>
  )
}

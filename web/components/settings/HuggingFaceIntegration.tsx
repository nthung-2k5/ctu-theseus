/**
 * Settings > Integrations > Hugging Face.
 *
 * A saved access token lets the user import models their Hugging Face account can read but the public cannot
 * (private repositories, and gated ones they have been granted). The server checks it with Hugging Face when it
 * is saved, stores it encrypted, and never sends it back: all this component ever shows is the account name and
 * the token's last characters. The token is only used to download models this same user imports.
 */

import { Anchor, Badge, Button, Code, Group, Paper, PasswordInput, Stack, Text, ThemeIcon } from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  CheckCircleIcon,
  CloudArrowDownIcon,
  LockSimpleIcon,
  PlugsConnectedIcon,
  PlugsIcon,
} from '@phosphor-icons/react'
import { QueryBoundary, SectionLabel } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  disconnectHuggingface,
  getConnectHuggingfaceMutationOptions,
  getGetHuggingfaceQueryKey,
  getGetHuggingfaceQueryOptions,
} from '@public/lib/api/generated/integrations/integrations'
import { formatDate } from '@public/lib/format'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

const TOKENS_URL = 'https://huggingface.co/settings/tokens'

/** What can be wrong with a token before Hugging Face is even asked; the server checks the rest. */
function tokenProblem(token: string): string | null {
  const t = token.trim()
  if (!t) return 'Paste your access token'
  if (!t.startsWith('hf_') || /\s/.test(t)) return 'A Hugging Face access token starts with hf_ and has no spaces'
  return null
}

export function HuggingFaceIntegration() {
  const queryClient = useQueryClient()
  const statusKey = getGetHuggingfaceQueryKey()
  const { data: status, isLoading, isError, refetch } = useQuery(getGetHuggingfaceQueryOptions())

  const [token, setToken] = useState('')
  const [replacing, setReplacing] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const connected = status?.connected === true
  const showForm = !connected || replacing

  const connect = useMutation({
    ...getConnectHuggingfaceMutationOptions(),
    onSuccess: (next) => {
      queryClient.setQueryData(statusKey, next)
      setToken('')
      setReplacing(false)
      notifications.show({
        title: 'Hugging Face connected',
        message: next.accountName ? `Signed in as ${next.accountName}.` : 'Your token was saved.',
        color: 'teal',
      })
    },
  })

  const disconnect = useMutation({
    mutationFn: async () => {
      await disconnectHuggingface()
    },
    onSuccess: () => {
      queryClient.setQueryData(statusKey, { connected: false })
      setReplacing(false)
      notifications.show({
        title: 'Hugging Face disconnected',
        message: 'Your token was deleted from this server.',
        color: 'gray',
      })
    },
    onError: (error) =>
      notifications.show({ title: 'Could not disconnect', message: apiErrorMessage(error), color: 'red' }),
  })

  const submit = () => {
    const problem = tokenProblem(token)
    setLocalError(problem)
    if (problem) return
    connect.mutate({ data: { token: token.trim() } })
  }

  const confirmDisconnect = () =>
    modals.openConfirmModal({
      title: 'Disconnect Hugging Face',
      children: (
        <Text size="sm">
          Delete your saved token? Private and gated models you already imported keep working until their files have to
          be downloaded again, which then needs a token.
        </Text>
      ),
      labels: { confirm: 'Disconnect', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: () => disconnect.mutate(),
    })

  const error = localError ?? (connect.isError ? apiErrorMessage(connect.error, 'Could not save the token') : null)

  return (
    <Paper p="md">
      <Stack gap="sm">
        <Group justify="space-between" align="flex-start" wrap="nowrap">
          <Group gap="sm" wrap="nowrap" align="flex-start">
            <ThemeIcon size={36} radius="md" variant="light" color={connected ? 'teal' : 'gray'}>
              <CloudArrowDownIcon size={20} />
            </ThemeIcon>
            <div>
              <SectionLabel>Hugging Face</SectionLabel>
              <Text size="xs" c="dimmed" maw={560}>
                Import private models, and gated ones you have been granted access to, by repository id. Only you can
                use your token, and only to download models you import.
              </Text>
            </div>
          </Group>
          {connected ? (
            <Badge color="teal" variant="light" leftSection={<CheckCircleIcon size={12} />}>
              Connected
            </Badge>
          ) : (
            <Badge color="gray" variant="light">
              Not connected
            </Badge>
          )}
        </Group>

        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {connected && (
            <Group justify="space-between" wrap="wrap" gap="sm">
              <div>
                <Text size="sm">
                  Signed in as <strong>{status?.accountName ?? 'your account'}</strong>
                </Text>
                <Text size="xs" c="dimmed" className="tnum">
                  Token <Code>hf_••••{status?.tokenHint}</Code>
                  {status?.connectedAt ? ` · saved ${formatDate(status.connectedAt)}` : ''}
                </Text>
              </div>
              {!replacing && (
                <Group gap="xs">
                  <Button variant="default" leftSection={<PlugsIcon size={14} />} onClick={() => setReplacing(true)}>
                    Replace token
                  </Button>
                  <Button
                    color="red"
                    variant="light"
                    loading={disconnect.isPending}
                    leftSection={<PlugsIcon size={14} />}
                    onClick={confirmDisconnect}
                  >
                    Disconnect
                  </Button>
                </Group>
              )}
            </Group>
          )}

          {showForm && (
            <form
              onSubmit={(e) => {
                e.preventDefault()
                submit()
              }}
            >
              <Stack gap="xs">
                <PasswordInput
                  label={replacing ? 'New access token' : 'Access token'}
                  placeholder="hf_…"
                  value={token}
                  onChange={(e) => {
                    setToken(e.currentTarget.value)
                    setLocalError(null)
                    if (connect.isError) connect.reset()
                  }}
                  error={error}
                  leftSection={<LockSimpleIcon size={14} />}
                  autoComplete="off"
                  spellCheck={false}
                  data-1p-ignore
                  description={
                    <>
                      Create a <strong>read</strong> token (or a fine-grained one limited to the repositories you want)
                      at{' '}
                      <Anchor href={TOKENS_URL} target="_blank" rel="noopener noreferrer" size="xs">
                        huggingface.co/settings/tokens
                      </Anchor>
                      . It is checked with Hugging Face, then stored encrypted and never shown again.
                    </>
                  }
                />
                <Group justify="flex-end" gap="xs">
                  {replacing && (
                    <Button
                      variant="subtle"
                      color="gray"
                      onClick={() => {
                        setReplacing(false)
                        setToken('')
                        setLocalError(null)
                        connect.reset()
                      }}
                    >
                      Cancel
                    </Button>
                  )}
                  <Button type="submit" leftSection={<PlugsConnectedIcon size={14} />} loading={connect.isPending}>
                    {replacing ? 'Save new token' : 'Connect'}
                  </Button>
                </Group>
              </Stack>
            </form>
          )}
        </QueryBoundary>
      </Stack>
    </Paper>
  )
}

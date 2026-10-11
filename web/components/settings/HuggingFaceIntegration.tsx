/**
 * Settings > Integrations > Hugging Face: a logo tile whose corner shows whether it is connected. Clicking it opens
 * a modal with the status and the actions (connect, replace the token, disconnect).
 *
 * A saved access token lets the user import models their Hugging Face account can read but the public cannot
 * (private repositories, and gated ones they have been granted). The server checks it with Hugging Face when it
 * is saved, stores it encrypted, and never sends it back: all this component ever shows is the account name and
 * the token's last characters. The token is only used to download models this same user imports.
 */

import {
  Anchor,
  Badge,
  Button,
  Code,
  Group,
  Image,
  Modal,
  Paper,
  PasswordInput,
  Stack,
  Text,
  Tooltip,
  UnstyledButton,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { CheckCircleIcon, CircleIcon, LockSimpleIcon, PlugsConnectedIcon, PlugsIcon } from '@phosphor-icons/react'
import HuggingFaceLogo from '@public/assets/huggingface.svg'
import { QueryBoundary } from '@public/components/ui'
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

  const [opened, { open, close }] = useDisclosure(false)
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

  const closeModal = () => {
    close()
    setReplacing(false)
    setToken('')
    setLocalError(null)
    connect.reset()
  }

  const statusLabel = isLoading ? 'checking…' : connected ? 'connected' : 'not connected'

  return (
    <div>
      <Tooltip label="Hugging Face">
        <UnstyledButton onClick={open} aria-label={`Hugging Face integration, ${statusLabel}`} w="fit-content">
          <Paper p="md" pos="relative" style={{ lineHeight: 0 }}>
            <Image src={HuggingFaceLogo} alt="" w={44} h={44} fit="contain" />
            <span style={{ position: 'absolute', top: 6, right: 6, display: 'flex' }}>
              {isLoading ? null : connected ? (
                <CheckCircleIcon size={16} weight="fill" color="var(--mantine-color-teal-6)" />
              ) : (
                <CircleIcon size={16} weight="bold" color="var(--mantine-color-dimmed)" />
              )}
            </span>
          </Paper>
        </UnstyledButton>
      </Tooltip>

      <Modal
        opened={opened}
        onClose={closeModal}
        centered
        title={
          <Group gap="xs" wrap="nowrap">
            <Image src={HuggingFaceLogo} alt="" w={24} h={24} fit="contain" flex="none" />
            <Text fw={600}>Hugging Face</Text>
            <Badge color={connected ? 'teal' : 'gray'} variant="light">
              {isLoading ? 'Checking…' : connected ? 'Connected' : 'Not connected'}
            </Badge>
          </Group>
        }
      >
        <Stack gap="sm">
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
                        Create a <strong>read</strong> token (or a fine-grained one limited to the repositories you
                        want) at{' '}
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
      </Modal>
    </div>
  )
}

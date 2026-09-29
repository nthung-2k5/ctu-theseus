/**
 * Admin > Users: find accounts, promote/demote admins, disable an account, sign a user out everywhere.
 * Every rule that protects the platform (no self-demotion, never the last admin) is enforced by the
 * API; the row buttons just surface its 400 message.
 */

import { Badge, Button, Group, Pagination, Select, Text, TextInput } from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { MagnifyingGlassIcon } from '@phosphor-icons/react'
import { DataTable, type DataTableColumn, PageHeader, QueryBoundary } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  adminRevokeUserSessions,
  adminUpdateUser,
  getAdminListUsersQueryKey,
  getAdminListUsersQueryOptions,
} from '@public/lib/api/generated/admin/admin'
import type { AdminUserOut, UpdateUserBody } from '@public/lib/api/generated/models'
import { sessionQueryOptions } from '@public/lib/auth'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

const PAGE_SIZE = 25

function confirm({
  title,
  message,
  label,
  color,
  onConfirm,
}: {
  title: string
  message: string
  label: string
  color?: string
  onConfirm: () => void
}) {
  modals.openConfirmModal({
    title,
    children: <Text size="sm">{message}</Text>,
    labels: { confirm: label, cancel: 'Cancel' },
    confirmProps: { color: color ?? 'primary' },
    onConfirm,
  })
}

export function UsersPage() {
  const queryClient = useQueryClient()
  const { data: me } = useQuery(sessionQueryOptions)

  const [search, setSearch] = useState('')
  const [role, setRole] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [q] = useDebouncedValue(search.trim(), 250)

  const params = { q: q || undefined, role: role || undefined, page, page_size: PAGE_SIZE }
  const { data, isLoading, isError, refetch } = useQuery({
    ...getAdminListUsersQueryOptions(params),
    placeholderData: (prev) => prev,
  })
  const users = data?.users ?? []
  const totalPages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE))

  const refresh = () => queryClient.invalidateQueries({ queryKey: getAdminListUsersQueryKey().slice(0, 1) })
  const onError = (error: unknown) =>
    notifications.show({ title: 'Error', message: apiErrorMessage(error), color: 'red' })

  const update = useMutation({
    mutationFn: ({ userId, body }: { userId: string; body: UpdateUserBody }) => adminUpdateUser(userId, body),
    onSuccess: refresh,
    onError,
  })
  const signOut = useMutation({
    mutationFn: (userId: string) => adminRevokeUserSessions(userId),
    onSuccess: () =>
      notifications.show({ title: 'Signed out', message: 'All of this user’s sessions were revoked.', color: 'gray' }),
    onError,
  })

  const columns: DataTableColumn<AdminUserOut>[] = [
    {
      key: 'user',
      header: 'User',
      render: (u) => (
        <div>
          <Text size="sm" fw={600}>
            {u.name}
          </Text>
          <Text size="xs" c="dimmed">
            {u.email}
          </Text>
        </div>
      ),
    },
    {
      key: 'role',
      header: 'Role',
      fit: true,
      render: (u) => (
        <Group gap={6}>
          <Badge variant="light" color={u.role === 'admin' ? 'grape' : 'gray'}>
            {u.role}
          </Badge>
          {u.disabled && (
            <Badge variant="light" color="red">
              disabled
            </Badge>
          )}
        </Group>
      ),
    },
    { key: 'projects', header: 'Projects', fit: true, render: (u) => <Text size="sm">{u.projectCount}</Text> },
    { key: 'runs', header: 'Runs', fit: true, render: (u) => <Text size="sm">{u.runCount}</Text> },
    { key: 'keys', header: 'API keys', fit: true, render: (u) => <Text size="sm">{u.apiKeyCount}</Text> },
    {
      key: 'created',
      header: 'Joined',
      fit: true,
      render: (u) => (
        <Text size="sm" c="dimmed">
          {new Date(u.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (u) => {
        const isMe = u.id === me?.id
        return (
          <Group gap={4} wrap="nowrap">
            <Button
              size="xs"
              variant="subtle"
              disabled={isMe && u.role === 'admin'}
              onClick={() =>
                confirm({
                  title: u.role === 'admin' ? 'Remove admin access' : 'Make admin',
                  message:
                    u.role === 'admin'
                      ? `${u.email} will lose access to the admin area.`
                      : `${u.email} will be able to manage every user, model and setting on the platform.`,
                  label: u.role === 'admin' ? 'Remove admin' : 'Make admin',
                  onConfirm: () =>
                    update.mutate({ userId: u.id, body: { role: u.role === 'admin' ? 'user' : 'admin' } }),
                })
              }
            >
              {u.role === 'admin' ? 'Demote' : 'Promote'}
            </Button>
            <Button
              size="xs"
              variant="subtle"
              color={u.disabled ? 'teal' : 'red'}
              disabled={isMe && !u.disabled}
              onClick={() =>
                u.disabled
                  ? update.mutate({ userId: u.id, body: { disabled: false } })
                  : confirm({
                      title: 'Disable account',
                      message: `${u.email} will be signed out and won’t be able to sign in. Their API keys are revoked. Their data is kept.`,
                      label: 'Disable',
                      color: 'red',
                      onConfirm: () => update.mutate({ userId: u.id, body: { disabled: true } }),
                    })
              }
            >
              {u.disabled ? 'Enable' : 'Disable'}
            </Button>
            <Button size="xs" variant="subtle" color="gray" disabled={isMe} onClick={() => signOut.mutate(u.id)}>
              Sign out
            </Button>
          </Group>
        )
      },
    },
  ]

  return (
    <div className="flex flex-col gap-2">
      <PageHeader title="Users" description={`${data?.total ?? 0} accounts`} />
      <Group gap="xs">
        <TextInput
          placeholder="Search name or email"
          leftSection={<MagnifyingGlassIcon size={14} />}
          value={search}
          onChange={(e) => {
            setSearch(e.currentTarget.value)
            setPage(1)
          }}
          w={280}
        />
        <Select
          placeholder="All roles"
          clearable
          data={[
            { value: 'admin', label: 'Admins' },
            { value: 'user', label: 'Users' },
          ]}
          value={role}
          onChange={(v) => {
            setRole(v)
            setPage(1)
          }}
          w={140}
        />
      </Group>

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        <DataTable columns={columns} data={users} getRowKey={(u) => u.id} emptyMessage="No matching users" />
        {totalPages > 1 && <Pagination value={page} onChange={setPage} total={totalPages} size="sm" />}
      </QueryBoundary>
    </div>
  )
}

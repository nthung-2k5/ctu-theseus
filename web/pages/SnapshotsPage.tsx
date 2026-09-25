/**
 * Snapshots ledger: a timeline of the project's immutable dataset snapshots. Each is a frozen,
 * versioned copy of the pool at the moment it was created; open one for its identity, split and items.
 */

import { Badge, Button, Group, Paper, Text, Timeline } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ArchiveIcon, GitCommitIcon, MagicWandIcon, PlusIcon, TrashIcon } from '@phosphor-icons/react'
import { splitCounts } from '@public/components/dataset/VersionBrowsing'
import { confirmDelete, EmptyState, LinkButton, PageHeader, StatusBadge } from '@public/components/ui'
import { deleteVersion as deleteVersionRequest } from '@public/lib/api/generated/datasets/datasets'
import { formatDateTime } from '@public/lib/format'
import { invalidateProjectScope, projectDetailQueryOptions } from '@public/lib/queries'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/snapshots/')

const VERSION_STATUS_COLORS: Record<string, string> = {
  building: 'cyan',
  ready: 'teal',
  failed: 'red',
}

export function SnapshotsPage() {
  const { projectId } = routeApi.useParams()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const dataset = project.dataset
  // Newest first, like a commit log.
  const versions = [...(dataset?.versions ?? [])].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  )

  const deleteVersion = useMutation({
    mutationFn: async (id: string) => {
      await deleteVersionRequest(id)
    },
    onSuccess: () => {
      invalidateProjectScope(queryClient, projectId)
      notifications.show({ title: 'Snapshot deleted', message: 'The snapshot has been removed', color: 'green' })
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to delete snapshot', color: 'red' })
    },
  })

  const handleDelete = (id: string) => {
    const version = versions.find((v) => v.id === id)
    confirmDelete({
      title: 'Delete snapshot',
      message: `Are you sure you want to delete "${version?.versionTag ?? 'this snapshot'}"? This cannot be undone. Any training run built from it keeps its own reference, but the snapshot itself will be gone.`,
      onConfirm: () => deleteVersion.mutate(id),
    })
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Snapshots"
        description={`Immutable dataset snapshots for "${project.name}"`}
        actions={
          <LinkButton
            to="/project/$projectId/snapshots/new"
            params={{ projectId }}
            leftSection={<PlusIcon size={14} />}
            disabled={!dataset?.draft}
          >
            New snapshot
          </LinkButton>
        }
      />

      {!dataset ? (
        <EmptyState
          icon={ArchiveIcon}
          title="No dataset found"
          description="This project's dataset hasn't been initialized yet."
        />
      ) : versions.length === 0 ? (
        <EmptyState
          icon={ArchiveIcon}
          title="No snapshots yet"
          description="Build one to freeze the current draft for training."
          action={
            <LinkButton
              to="/project/$projectId/snapshots/new"
              params={{ projectId }}
              leftSection={<PlusIcon size={14} />}
            >
              New snapshot
            </LinkButton>
          }
        />
      ) : (
        <Paper p="md">
          <Timeline bulletSize={22} lineWidth={2} active={versions.length}>
            {versions.map((version, index) => {
              const counts = splitCounts(version)
              return (
                <Timeline.Item
                  key={version.id}
                  bullet={<GitCommitIcon size={14} />}
                  title={
                    <Group gap="xs" wrap="wrap">
                      <Text fw={600} size="sm">
                        {version.versionTag ?? 'Untitled snapshot'}
                      </Text>
                      {index === 0 && <Badge color="cyan">latest</Badge>}
                      <StatusBadge value={version.status} colorMap={VERSION_STATUS_COLORS} />
                      {version.augmentedCount > 0 && (
                        <Badge color="grape" leftSection={<MagicWandIcon size={10} />}>
                          +{version.augmentedCount} augmented
                        </Badge>
                      )}
                    </Group>
                  }
                >
                  <Text size="xs" c="dimmed" className="tnum">
                    {formatDateTime(version.createdAt)} · {version.itemCount ?? 0} items · train {counts.train} · val{' '}
                    {counts.validation} · test {counts.test}
                  </Text>
                  <Group gap="xs" mt={6}>
                    <LinkButton
                      to="/project/$projectId/snapshots/$versionId"
                      params={{ projectId, versionId: version.id }}
                      search={{ page: 1, perPage: 20 }}
                      variant="light"
                      size="compact-sm"
                    >
                      Open details
                    </LinkButton>
                    <Button
                      size="compact-sm"
                      variant="subtle"
                      color="red"
                      leftSection={<TrashIcon size={12} />}
                      loading={deleteVersion.isPending && deleteVersion.variables === version.id}
                      onClick={() => handleDelete(version.id)}
                    >
                      Delete
                    </Button>
                  </Group>
                </Timeline.Item>
              )
            })}
          </Timeline>
        </Paper>
      )}
    </div>
  )
}

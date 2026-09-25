/**
 * One snapshot: identity, split and augmentation summary, then its items. Split assignments and item
 * membership can no longer change here, only be viewed per split, or the whole snapshot deleted.
 */

import { Alert, Badge, Button, Group, Paper, SimpleGrid, Stack, Table, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ArrowLeftIcon, MagicWandIcon, TrashIcon } from '@phosphor-icons/react'
import { ItemsByModality } from '@public/components/dataset/ItemsByModality'
import { type ItemOrigin, type ItemSort, ItemsFilterBar } from '@public/components/dataset/ItemsFilterBar'
import { ItemsPaginationBar } from '@public/components/dataset/ItemsPaginationBar'
import { SPLIT_TYPES, splitCounts } from '@public/components/dataset/VersionBrowsing'
import {
  CopyField,
  confirmDelete,
  EmptyState,
  LinkButton,
  PageHeader,
  ProportionBar,
  SectionLabel,
  StatCard,
  StatusBadge,
} from '@public/components/ui'
import { deleteVersion as deleteVersionRequest } from '@public/lib/api/generated/datasets/datasets'
import { SPLIT_COLORS } from '@public/lib/constants'
import { formatDateTime } from '@public/lib/format'
import {
  invalidateProjectScope,
  projectDetailQueryOptions,
  useLabelClasses,
  useProjectItems,
} from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/snapshots/$versionId')

/** "image_horizontal_flip" -> "horizontal flip": the op ids are `<modality>_<name>`. */
const augmentationLabel = (id: string) => id.split('_').slice(1).join(' ')

const VERSION_STATUS_COLORS: Record<string, string> = { building: 'cyan', ready: 'teal', failed: 'red' }

const cap = (s: string) => `${s[0].toUpperCase()}${s.slice(1)}`

export function SnapshotDetailPage() {
  const { projectId, versionId } = routeApi.useParams()
  const { page, perPage, split, classId, search, sort, origin } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(project.task)
  const versions = project.dataset?.versions ?? []
  const version = versions.find((v) => v.id === versionId) ?? null
  const { data: classesData } = useLabelClasses(descriptor.annotation.requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []
  const classNameById = new Map(classes.map((c) => [c.classId, c.name]))

  const { data, isLoading, isError, refetch } = useProjectItems(version ? projectId : undefined, {
    versionId,
    split: split ?? undefined,
    classId: classId ?? undefined,
    search: search || undefined,
    page,
    perPage,
    sort: sort ?? 'newest',
    origin: version && version.augmentedCount > 0 ? (origin ?? undefined) : undefined,
  })
  const items = data?.items ?? []
  const total = data?.total ?? 0

  const goToList = () => navigate({ to: '/project/$projectId/snapshots', params: { projectId } })

  const deleteVersion = useMutation({
    mutationFn: async () => {
      await deleteVersionRequest(versionId)
    },
    onSuccess: () => {
      invalidateProjectScope(queryClient, projectId)
      notifications.show({ title: 'Snapshot deleted', message: 'The snapshot has been removed', color: 'green' })
      void goToList()
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to delete snapshot', color: 'red' }),
  })

  if (!version) {
    return (
      <div className="flex flex-col gap-3 p-3">
        <EmptyState
          title="Snapshot not found"
          description="It may have been deleted."
          action={
            <LinkButton to="/project/$projectId/snapshots" params={{ projectId }} variant="default">
              Back to snapshots
            </LinkButton>
          }
        />
      </div>
    )
  }

  const counts = splitCounts(version)
  const config = version.augmentationConfig
  const setSearch = (patch: Record<string, unknown>) => navigate({ search: (prev) => ({ ...prev, ...patch, page: 1 }) })

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title={version.versionTag ?? 'Untitled snapshot'}
        description={`Snapshot of "${project.name}"`}
        badges={
          <>
            <StatusBadge value={version.status} colorMap={VERSION_STATUS_COLORS} />
            {version.augmentedCount > 0 && (
              <Badge color="grape" leftSection={<MagicWandIcon size={10} />}>
                augmented
              </Badge>
            )}
          </>
        }
        actions={
          <>
            <Button variant="default" leftSection={<ArrowLeftIcon size={14} />} onClick={goToList}>
              Back
            </Button>
            <Button
              variant="light"
              color="red"
              leftSection={<TrashIcon size={14} />}
              loading={deleteVersion.isPending}
              onClick={() =>
                confirmDelete({
                  title: 'Delete snapshot',
                  message: `Delete "${version.versionTag ?? 'this snapshot'}"? This cannot be undone. Any training run built from it keeps its own reference, but the snapshot itself will be gone.`,
                  onConfirm: () => deleteVersion.mutate(),
                })
              }
            >
              Delete
            </Button>
          </>
        }
      />

      <SimpleGrid cols={{ base: 2, md: 4 }} spacing="sm">
        <StatCard label="Items" value={version.itemCount ?? 0} />
        <StatCard label="Classes" value={version.classCount ?? '—'} />
        <StatCard label="Augmented" value={version.augmentedCount} />
        <StatCard label="Built" value={version.builtAt ? formatDateTime(version.builtAt) : '—'} />
      </SimpleGrid>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
        <Paper p="md">
          <Stack gap="xs">
            <SectionLabel>Identity</SectionLabel>
            <Group gap="xs" justify="space-between" wrap="nowrap">
              <Text size="xs" c="dimmed">
                Snapshot id
              </Text>
              <CopyField value={version.id} />
            </Group>
            <Group gap="xs" justify="space-between">
              <Text size="xs" c="dimmed">
                Created
              </Text>
              <Text size="xs" className="tnum">
                {formatDateTime(version.createdAt)}
              </Text>
            </Group>
            {version.failedMessage && (
              <Alert color="red" p="xs" title="Build failed">
                {version.failedMessage}
              </Alert>
            )}
          </Stack>
        </Paper>

        <Paper p="md">
          <Stack gap="xs">
            <SectionLabel>Split</SectionLabel>
            <ProportionBar
              height={20}
              segments={SPLIT_TYPES.map((s) => ({
                key: s,
                label: cap(s),
                value: counts[s],
                color: SPLIT_COLORS[s] ?? 'gray',
              }))}
            />
            <Table verticalSpacing={2} withRowBorders={false} className="tnum">
              <Table.Tbody>
                {SPLIT_TYPES.map((s) => (
                  <Table.Tr key={s}>
                    <Table.Td>
                      <Group gap={6} wrap="nowrap">
                        <span style={{ width: 8, height: 8, background: SPLIT_COLORS[s] }} />
                        <Text size="xs">{cap(s)}</Text>
                      </Group>
                    </Table.Td>
                    <Table.Td ta="right">
                      <Text size="xs">{counts[s]}</Text>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Stack>
        </Paper>
      </SimpleGrid>

      {config && (
        <Paper p="md">
          <Stack gap="xs">
            <SectionLabel>Augmentation</SectionLabel>
            <Alert icon={<MagicWandIcon size={16} />} color="grape" p="xs">
              {version.augmentedCount} augmented item{version.augmentedCount === 1 ? '' : 's'} were added to the train
              split ({config.copiesPerItem ?? 1} cop{(config.copiesPerItem ?? 1) === 1 ? 'y' : 'ies'} per item).
              Validation and test items are original.
              {config.skippedItems ? ` ${config.skippedItems} item(s) could not be augmented and were skipped.` : ''}
            </Alert>
            <Table verticalSpacing={4} className="tnum">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Operation</Table.Th>
                  <Table.Th ta="right">Probability</Table.Th>
                  <Table.Th>Parameters</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {config.ops.map((op) => (
                  <Table.Tr key={op.id}>
                    <Table.Td tt="capitalize">{augmentationLabel(op.id)}</Table.Td>
                    <Table.Td ta="right">{Math.round((op.probability ?? 1) * 100)}%</Table.Td>
                    <Table.Td>
                      <Text size="xs" c="dimmed">
                        {Object.entries(op.params ?? {})
                          .map(([k, v]) => `${k}=${String(v)}`)
                          .join(', ') || '—'}
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Stack>
        </Paper>
      )}

      <Stack gap="sm">
        <SectionLabel>Items</SectionLabel>
        <ItemsFilterBar
          version={version}
          split={split ?? null}
          onSplitChange={(s) => setSearch({ split: s ?? undefined })}
          classes={classes}
          classId={classId ?? null}
          onClassChange={(c) => setSearch({ classId: c ?? undefined })}
          search={search ?? ''}
          onSearchChange={(s) => setSearch({ search: s || undefined })}
          sort={(sort ?? 'newest') as ItemSort}
          onSortChange={(s) => setSearch({ sort: s })}
          origin={(origin ?? null) as ItemOrigin | null}
          onOriginChange={(o) => setSearch({ origin: o ?? undefined })}
        />
        <ItemsByModality
          items={items}
          modality={descriptor.modality}
          isLoading={isLoading}
          isError={isError}
          onRetry={() => refetch()}
          classNameById={classNameById}
          emptyMessage="No items in this split"
        />
        {items.length > 0 && (
          <ItemsPaginationBar
            total={total}
            page={page}
            perPage={perPage}
            onPageChange={(p) => navigate({ search: (prev) => ({ ...prev, page: p }) })}
            onPerPageChange={(pp) => navigate({ search: (prev) => ({ ...prev, perPage: pp, page: 1 }) })}
          />
        )}
      </Stack>
    </div>
  )
}

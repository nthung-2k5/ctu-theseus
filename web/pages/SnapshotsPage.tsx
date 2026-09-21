/**
 * Snapshots page – browse the project's immutable dataset snapshots. Each
 * snapshot is a frozen, versioned copy of the pool at the moment it was
 * created (see the Dataset page's "Create Snapshot") — split assignments
 * and item membership can no longer change here, only be viewed per split,
 * or the whole snapshot deleted.
 */

import { Badge, Box, Button, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ArchiveIcon, ArrowLeftIcon, TrashIcon } from '@phosphor-icons/react'
import { ItemsByModality } from '@public/components/dataset/ItemsByModality'
import { type ItemSort, ItemsFilterBar } from '@public/components/dataset/ItemsFilterBar'
import { ItemsPaginationBar } from '@public/components/dataset/ItemsPaginationBar'
import { SPLIT_TYPES, SplitProgressBar, splitCounts } from '@public/components/dataset/VersionBrowsing'
import { confirmDelete, DataTable, type DataTableColumn, EmptyState, PageHeader } from '@public/components/ui'
import { deleteVersion as deleteVersionRequest } from '@public/lib/api/generated/datasets/datasets'
import { SPLIT_CHART_COLORS } from '@public/lib/constants'
import {
  invalidateProjectScope,
  projectDetailQueryOptions,
  useLabelClasses,
  useProjectItems,
} from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import type { DatasetVersion } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/snapshots')

/* ── Items viewer for the selected snapshot/split ── */
const SnapshotItemsPanel = ({
  projectId,
  versionId,
  splitType,
  classId,
  search,
  modality,
  classNameById,
  page,
  perPage,
  sort,
  onPageChange,
  onPerPageChange,
}: {
  projectId: string
  versionId: string
  splitType: (typeof SPLIT_TYPES)[number] | null
  classId: string | null
  search: string
  modality: ReturnType<typeof getTaskDescriptor>['modality']
  classNameById: Map<string, string>
  page: number
  perPage: number
  sort: ItemSort
  onPageChange: (page: number) => void
  onPerPageChange: (perPage: number) => void
}) => {
  const { data, isLoading, isError, refetch } = useProjectItems(projectId, {
    versionId,
    split: splitType ?? undefined,
    classId: classId ?? undefined,
    search: search || undefined,
    page,
    perPage,
    sort,
  })
  const items = data?.items ?? []
  const total = data?.total ?? 0

  return (
    <Stack gap="md">
      <ItemsByModality
        items={items}
        modality={modality}
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
          onPageChange={onPageChange}
          onPerPageChange={onPerPageChange}
        />
      )}
    </Stack>
  )
}

/* ── Main Snapshots page ── */
export function SnapshotsPage() {
  const { projectId } = routeApi.useParams()
  const { page, perPage, versionId, split, classId, search, sort } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const queryClient = useQueryClient()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const dataset = activeProject.dataset
  const descriptor = getTaskDescriptor(activeProject.task)
  const versions = dataset?.versions ?? []
  const { data: classesData } = useLabelClasses(descriptor.annotation.requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []
  const classNameById = new Map(classes.map((c) => [c.classId, c.name]))

  const selectedVersion = versionId ? (versions.find((v) => v.id === versionId) ?? null) : null

  const goToList = () => navigate({ search: (prev) => ({ ...prev, versionId: undefined, split: undefined, page: 1 }) })
  const selectVersion = (nextVersionId: string) =>
    navigate({ search: (prev) => ({ ...prev, versionId: nextVersionId, split: undefined, page: 1 }) })
  const selectSplit = (nextSplit: (typeof SPLIT_TYPES)[number] | null) =>
    navigate({ search: (prev) => ({ ...prev, split: nextSplit ?? undefined, page: 1 }) })
  const selectClass = (nextClassId: string | null) =>
    navigate({ search: (prev) => ({ ...prev, classId: nextClassId ?? undefined, page: 1 }) })
  const selectSearch = (nextSearch: string) =>
    navigate({ search: (prev) => ({ ...prev, search: nextSearch || undefined, page: 1 }) })
  const selectSort = (nextSort: ItemSort) => navigate({ search: (prev) => ({ ...prev, sort: nextSort, page: 1 }) })

  const deleteVersion = useMutation({
    mutationFn: async (id: string) => {
      await deleteVersionRequest(id)
    },
    onSuccess: (_data, id) => {
      invalidateProjectScope(queryClient, projectId)
      notifications.show({ title: 'Snapshot deleted', message: 'The snapshot has been removed', color: 'green' })
      if (versionId === id) goToList()
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to delete snapshot', color: 'red' })
    },
  })

  const handleDeleteVersion = (id: string) => {
    const version = versions.find((v) => v.id === id)
    confirmDelete({
      title: 'Delete snapshot',
      message: `Are you sure you want to delete "${version?.versionTag ?? 'this snapshot'}"? This cannot be undone — any training run built from it will keep its own reference, but the snapshot itself will be gone.`,
      onConfirm: () => deleteVersion.mutate(id),
    })
  }

  const versionColumns: DataTableColumn<DatasetVersion>[] = [
    {
      key: 'versionTag',
      header: 'Snapshot',
      render: (version) => (
        <Text size="sm" fw={600}>
          {version.versionTag ?? 'Untitled snapshot'}
        </Text>
      ),
    },
    ...SPLIT_TYPES.map((splitType) => ({
      key: splitType,
      header: `${splitType[0].toUpperCase()}${splitType.slice(1)}`,
      fit: true,
      render: (version: DatasetVersion) => (
        <Badge size="sm" color={SPLIT_CHART_COLORS[splitType] ?? 'gray'}>
          {splitCounts(version)[splitType]}
        </Badge>
      ),
    })),
    {
      key: 'items',
      header: 'Items',
      fit: true,
      render: (version) => (
        <Text size="sm" c="dimmed">
          {version.itemCount ?? 0} items
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (version) => (
        <Text size="sm" c="dimmed">
          {new Date(version.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (version) => (
        <Button
          size="xs"
          variant="subtle"
          color="red"
          leftSection={<TrashIcon size={14} />}
          loading={deleteVersion.isPending && deleteVersion.variables === version.id}
          onClick={(e) => {
            e.stopPropagation()
            handleDeleteVersion(version.id)
          }}
        >
          Delete
        </Button>
      ),
    },
  ]

  /* ── Snapshot detail page ── */
  if (selectedVersion) {
    return (
      <Box mb="-1.25rem">
        <Stack gap="xl">
          <PageHeader
            title={selectedVersion.versionTag ?? 'Untitled snapshot'}
            description={`Snapshot of "${activeProject.name}"`}
            actions={
              <Button variant="subtle" color="gray" leftSection={<ArrowLeftIcon size={16} />} onClick={goToList}>
                Back to snapshots
              </Button>
            }
          />

          <Stack gap="md">
            <Stack gap="sm">
              <SplitProgressBar version={selectedVersion} />
              <ItemsFilterBar
                version={selectedVersion}
                split={split ?? null}
                onSplitChange={selectSplit}
                classes={classes}
                classId={classId ?? null}
                onClassChange={selectClass}
                search={search ?? ''}
                onSearchChange={selectSearch}
                sort={sort ?? 'newest'}
                onSortChange={selectSort}
              />
            </Stack>
            <SnapshotItemsPanel
              projectId={projectId}
              versionId={selectedVersion.id}
              splitType={split ?? null}
              classId={classId ?? null}
              search={search ?? ''}
              modality={descriptor.modality}
              classNameById={classNameById}
              page={page}
              perPage={perPage}
              sort={sort ?? 'newest'}
              onPageChange={(page) => navigate({ search: (prev) => ({ ...prev, page }) })}
              onPerPageChange={(perPage) => navigate({ search: (prev) => ({ ...prev, perPage, page: 1 }) })}
            />
          </Stack>
        </Stack>
      </Box>
    )
  }

  /* ── Snapshot list page ── */
  return (
    <Box>
      <Stack gap="xl">
        <PageHeader title="Snapshots" description={`Immutable dataset snapshots for "${activeProject.name}"`} />

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
            description="Create one from the Dataset page to freeze the current draft for training."
          />
        ) : (
          <DataTable
            columns={versionColumns}
            data={versions}
            getRowKey={(v) => v.id}
            onRowClick={(v) => selectVersion(v.id)}
          />
        )}
      </Stack>
    </Box>
  )
}

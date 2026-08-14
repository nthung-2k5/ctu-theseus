import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Group,
  Modal,
  Pagination,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { DatabaseIcon, PlusIcon, StackIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { LabelingProgress } from '@public/components/dataset/LabelingProgress'
import { DataTable, type DataTableColumn, EmptyState, PageHeader, StatusBadge } from '@public/components/ui'
import { useEden } from '@public/lib/api'
import { MODALITY_COLORS, SPLIT_COLORS } from '@public/lib/constants'
import { projectDetailQueryOptions, useProjectItems } from '@public/lib/queries'
import type { DatasetVersion } from '@public/store/types'
import { getTaskDescriptor } from '@server/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/dataset')

/* ── Create Version Modal ── */
const CreateVersionModal = ({
  projectId,
  needsAnnotations,
  onClose,
}: {
  projectId: string
  needsAnnotations: boolean
  onClose: () => void
}) => {
  const form = useForm({
    initialValues: { versionTag: '' },
    validate: {
      versionTag: (v) => (v.trim().length > 0 ? null : 'Version tag is required'),
    },
  })

  const eden = useEden()
  const queryClient = useQueryClient()

  // Cheap way to get the draft's total/labeledCount without fetching every
  // row — the aggregate counts are computed server-side regardless of
  // perPage (see GET /projects/:projectId/items).
  const { data: draftCounts } = useProjectItems(needsAnnotations ? projectId : undefined, { perPage: 1 })
  const total = draftCounts?.total ?? 0
  const labeledCount = draftCounts?.labeledCount ?? 0
  const unlabeledCount = total - labeledCount

  const createVersion = useMutation({
    ...eden.api.projects({ projectId }).versions.post.mutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      notifications.show({ title: 'Version created', message: 'New snapshot version created', color: 'green' })
      form.reset()
      onClose()
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to create version', color: 'red' })
    },
  })

  return (
    <form onSubmit={form.onSubmit((values) => createVersion.mutate(values))}>
      <Stack gap="md">
        {needsAnnotations && unlabeledCount > 0 && (
          <Alert icon={<WarningCircleIcon size={16} />} color="yellow" title="Unlabeled items in the draft">
            {unlabeledCount} of {total} items have no label yet — they'll snapshot with a null label column and won't
            contribute a usable training signal. Label them on the Data page before snapshotting, or continue anyway.
          </Alert>
        )}
        <TextInput
          label="Version tag"
          placeholder="e.g. v1.0, snapshot-2024-01"
          {...form.getInputProps('versionTag')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" loading={createVersion.isPending}>
            Create Snapshot
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

/* ── Split Items Viewer ── */
const SplitItemsPanel = ({
  projectId,
  versionId,
  splitType,
  page,
  onPageChange,
}: {
  projectId: string
  versionId: string
  splitType: 'train' | 'validation' | 'test'
  page: number
  onPageChange: (page: number) => void
}) => {
  const perPage = 20

  const { data, isLoading } = useProjectItems(projectId, { versionId, split: splitType, page, perPage })
  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  type SplitItem = (typeof items)[number]
  const columns: DataTableColumn<SplitItem>[] = [
    {
      key: 'id',
      header: 'ID',
      render: (item) => (
        <Text size="xs" ff="monospace">
          {item.id.slice(0, 8)}…
        </Text>
      ),
    },
    {
      key: 'externalId',
      header: 'External ID',
      render: (item) => <Text size="xs">{item.externalId ?? '—'}</Text>,
    },
    {
      key: 'features',
      header: 'Features',
      render: (item) => (
        <Group gap={4}>
          {item.textFeatures && (
            <Badge size="xs" color="blue">
              text
            </Badge>
          )}
          {item.visionFeatures && (
            <Badge size="xs" color="green">
              vision
            </Badge>
          )}
          {item.audioFeatures && (
            <Badge size="xs" color="orange">
              audio
            </Badge>
          )}
          {item.tabularFeatures && (
            <Badge size="xs" color="grape">
              tabular
            </Badge>
          )}
        </Group>
      ),
    },
    {
      key: 'annotations',
      header: 'Annotations',
      render: (item) => (
        <Badge size="xs" variant="light">
          {item.annotations?.length ?? 0}
        </Badge>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      render: (item) => (
        <Text size="xs" c="dimmed">
          {new Date(item.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
  ]

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text size="sm" fw={600} tt="capitalize">
          {splitType} Split
        </Text>
        <Badge variant="light" size="sm">
          {total} items
        </Badge>
      </Group>

      <DataTable
        columns={columns}
        data={items}
        getRowKey={(item) => item.id}
        loading={isLoading}
        emptyMessage="No items in this split yet"
      />

      {totalPages > 1 && (
        <Group justify="center">
          <Pagination total={totalPages} value={page} onChange={onPageChange} size="sm" />
        </Group>
      )}
    </Stack>
  )
}

const SPLIT_TYPES = ['train', 'validation', 'test'] as const

/* ── Version Card ── */
const VersionCard = ({
  version,
  isDraft,
  onSelectSplit,
}: {
  version: DatasetVersion
  isDraft: boolean
  onSelectSplit: (versionId: string, splitType: (typeof SPLIT_TYPES)[number]) => void
}) => {
  const counts = Object.fromEntries(SPLIT_TYPES.map((s) => [s, 0])) as Record<(typeof SPLIT_TYPES)[number], number>
  for (const item of version.items ?? []) counts[item.splitType]++

  return (
    <Card withBorder p="lg" radius="md">
      <Group justify="space-between" mb="md">
        <Group gap="sm">
          <ThemeIcon size="md" variant="light" color={isDraft ? 'blue' : 'teal'}>
            <StackIcon size={16} />
          </ThemeIcon>
          <Title order={5}>{isDraft ? 'Draft' : version.versionTag}</Title>
          <Badge variant="light" color={isDraft ? 'blue' : 'teal'} size="sm">
            {isDraft ? 'Working Copy' : 'Snapshot'}
          </Badge>
        </Group>
        <Text size="xs" c="dimmed">
          {new Date(version.createdAt).toLocaleDateString()}
        </Text>
      </Group>

      <SimpleGrid cols={3} spacing="sm">
        {SPLIT_TYPES.map((splitType) => (
          <Card
            key={splitType}
            withBorder
            p="sm"
            radius="sm"
            className="card-elevated"
            style={{ cursor: 'pointer' }}
            onClick={() => onSelectSplit(version.id, splitType)}
          >
            <Group justify="space-between">
              <Text size="sm" fw={500} tt="capitalize">
                {splitType}
              </Text>
              <Badge size="xs" variant="light" color={SPLIT_COLORS[splitType] ?? 'gray'}>
                {counts[splitType]}
              </Badge>
            </Group>
          </Card>
        ))}
      </SimpleGrid>
    </Card>
  )
}

/* ── Main Dataset Page ── */
export function DatasetPage() {
  const { projectId } = routeApi.useParams()
  const { page, versionId, split } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery({
    ...projectDetailQueryOptions(projectId),
    // Snapshot builds are async (see server/lib/snapshot.ts) — poll while
    // any version is still building so 'building' -> 'ready' shows up
    // without a manual refresh.
    refetchInterval: (query) => {
      const dataset = query.state.data?.project.dataset
      const versions = [dataset?.draft, ...(dataset?.versions ?? [])]
      return versions.some((v) => v?.status === 'building') ? 3000 : false
    },
  })
  const dataset = activeProject.dataset
  const needsAnnotations = getTaskDescriptor(activeProject.task).columns.some((c) => c.kind === 'label')

  const [createVersionOpened, { open: openCreateVersion, close: closeCreateVersion }] = useDisclosure(false)
  const selectedSplit = versionId && split ? { versionId, type: split } : null
  const setSelectedSplit = (next: { versionId: string; type: 'train' | 'validation' | 'test' } | null) =>
    navigate({ search: (prev) => ({ ...prev, versionId: next?.versionId, split: next?.type, page: 1 }) })

  // Cheap aggregate-only fetch (perPage=1) for the draft's labeling progress.
  const { data: draftCounts } = useProjectItems(needsAnnotations ? projectId : undefined, { perPage: 1 })

  return (
    <Box>
      <Stack gap="xl">
        <PageHeader
          title="Dataset"
          description={`Manage dataset for "${activeProject.name}"`}
          actions={
            dataset && (
              <Group gap="sm">
                {needsAnnotations && draftCounts && (
                  <LabelingProgress labeled={draftCounts.labeledCount} total={draftCounts.total} />
                )}
                <StatusBadge value={dataset.modality} colorMap={MODALITY_COLORS} />
                <Button size="xs" leftSection={<PlusIcon size={14} />} variant="light" onClick={openCreateVersion}>
                  Create Snapshot
                </Button>
              </Group>
            )
          }
        />

        {!dataset ? (
          <EmptyState
            icon={DatabaseIcon}
            title="No dataset found"
            description="This project's dataset hasn't been initialized yet."
          />
        ) : (
          <Stack gap="lg">
            {/* Draft version */}
            {dataset.draft && (
              <VersionCard
                version={dataset.draft}
                isDraft
                onSelectSplit={(versionId, type) => setSelectedSplit({ versionId, type })}
              />
            )}

            {/* Snapshot versions */}
            {(dataset.versions?.length ?? 0) > 0 && (
              <div>
                <Title order={5} mb="md">
                  Snapshots
                </Title>
                <Stack gap="sm">
                  {dataset.versions.map((version) => (
                    <VersionCard
                      key={version.id}
                      version={version}
                      isDraft={false}
                      onSelectSplit={(versionId, type) => setSelectedSplit({ versionId, type })}
                    />
                  ))}
                </Stack>
              </div>
            )}

            {(dataset.versions?.length ?? 0) === 0 && (
              <EmptyState
                description="No snapshots yet. Create one to freeze the current dataset state for training."
                compact
              />
            )}

            {/* Selected split items */}
            {selectedSplit && (
              <Card withBorder p="lg" radius="md">
                <Group justify="space-between" mb="md">
                  <Title order={5}>Items</Title>
                  <Button size="xs" variant="subtle" onClick={() => setSelectedSplit(null)}>
                    Close
                  </Button>
                </Group>
                <SplitItemsPanel
                  projectId={projectId}
                  versionId={selectedSplit.versionId}
                  splitType={selectedSplit.type}
                  page={page}
                  onPageChange={(page) => navigate({ search: (prev) => ({ ...prev, page }) })}
                />
              </Card>
            )}
          </Stack>
        )}
      </Stack>

      {/* Create Version Modal */}
      <Modal opened={createVersionOpened} onClose={closeCreateVersion} title="Create Snapshot Version" centered>
        <CreateVersionModal projectId={projectId} needsAnnotations={needsAnnotations} onClose={closeCreateVersion} />
      </Modal>
    </Box>
  )
}

import {
  Badge,
  Box,
  Button,
  Card,
  Center,
  Group,
  Pagination,
  Popover,
  SegmentedControl,
  Select,
  SimpleGrid,
  Skeleton,
  Slider,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  FunnelSimpleIcon,
  GridFourIcon,
  ImagesIcon,
  ListIcon,
  MagicWandIcon,
  SortAscendingIcon,
} from '@phosphor-icons/react'
import { GridCard } from '@public/components/dataset/GridCard'
import { ListRow } from '@public/components/dataset/ListRow'
import { SelectionToolbar } from '@public/components/dataset/SelectionToolbar'
import { api } from '@public/lib/api'
import { useEdenMutation } from '@public/lib/eden-query'
import { queries } from '@public/queries'
import {
  type DatasetSearchQuery,
  useProjectClasses,
  useProjectDataset,
  useProjectDatasetStats,
} from '@public/queries/project'
import type { DatasetSplitValue } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { useMemo, useState } from 'react'
import { useParams } from 'wouter'

type SortMode = 'none' | 'newest' | 'oldest' | 'filename'
type ViewMode = 'grid' | 'list'

const SPLIT_FILTER_OPTIONS = [
  { value: 'train' as const, label: 'Train' },
  { value: 'validation' as const, label: 'Validation' },
  { value: 'test' as const, label: 'Test' },
]

const SORT_OPTIONS = [
  { value: 'none', label: 'Default' },
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'filename', label: 'Filename A-Z' },
]

const PER_PAGE_OPTIONS = [
  { value: '20', label: '20' },
  { value: '30', label: '30' },
  { value: '50', label: '50' },
  { value: '100', label: '100' },
]

export function DatasetPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id
  const activeProject = useProjectStore((s) => s.activeProject)

  /* ── State ── */
  const [searchTerm, setSearchTerm] = useState('')
  const [filterClass, setFilterClass] = useState<string | null>(null)
  const [filterSplit, setFilterSplit] = useState<DatasetSplitValue | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [sortMode, setSortMode] = useState<SortMode>('none')
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(30)

  /* Auto-split percentages */
  const [splitTrain, setSplitTrain] = useState(70)
  const [splitVal, setSplitVal] = useState(20)
  const [splitTest, setSplitTest] = useState(10)

  /* Debounce search term for API calls */
  const [debouncedSearch] = useDebouncedValue(searchTerm, 300)

  /* ── Build query params ── */
  const queryParams = useMemo(() => {
    const q: DatasetSearchQuery = {
      page,
      perPage,
      search: debouncedSearch,
      sort: sortMode,
    }

    if (filterClass !== null) q.classId = filterClass
    if (filterSplit !== null) q.split = filterSplit
    return q
  }, [page, perPage, debouncedSearch, filterClass, filterSplit, sortMode])

  /* ── Queries ── */
  const { data: imgData, isLoading } = useProjectDataset(projectId, queryParams)
  const images = imgData?.images ?? []
  const totalImages = imgData?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(totalImages / perPage))

  /* ── Stats (independent of pagination) ── */
  const { data: statsData } = useProjectDatasetStats(projectId)
  const splitStats = useMemo(
    () =>
      statsData ?? {
        train: 0,
        validation: 0,
        test: 0,
        unassigned: 0,
        total: 0,
      },
    [statsData],
  )

  const { data: classData } = useProjectClasses(projectId)
  const classes = classData?.classes ?? []

  /* ── Mutations ── */
  const updateImage = useEdenMutation(
    ({
      imageId,
      classId,
      split,
    }: {
      imageId: string
      classId: string | null | undefined
      split: DatasetSplitValue | null | undefined
    }) => api.images({ imageId }).patch({ classId, split }),
    [queries.projects.detail(projectId)._ctx.dataset.queryKey],
    {
      onError: () => notifications.show({ title: 'Error', message: 'Failed to update image', color: 'red' }),
    },
  )

  const deleteImageById = useEdenMutation(
    (imageId: string) => api.images({ imageId }).delete(),
    [queries.projects.detail(projectId)._ctx.dataset.queryKey],
    {
      onSuccess: (imageId: string) => {
        setSelectedIds((prev) => {
          const next = new Set(prev)
          next.delete(imageId)
          return next
        })
      },
      onError: () => notifications.show({ title: 'Error', message: 'Failed to delete image', color: 'red' }),
    },
  )

  const autoSplit = useEdenMutation(
    (body: { train: number; validation: number; test: number }) =>
      api.projects({ projectId }).images['auto-split'].post(body),
    [queries.projects.detail(projectId)._ctx.dataset.queryKey],
    {
      onSuccess: (data) => {
        notifications.show({
          title: 'Auto-split complete',
          message: `Train: ${data.counts.train}, Validation: ${data.counts.validation}, Test: ${data.counts.test}`,
          color: 'green',
        })
      },
      onError: () => notifications.show({ title: 'Error', message: 'Auto-split failed', color: 'red' }),
    },
  )

  /* ── Selection helpers ── */
  const toggleSelect = (id: string, checked: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      checked ? next.add(id) : next.delete(id)
      return next
    })
  }

  const selectAll = () => setSelectedIds(new Set(images.map((i) => i.id)))
  const deselectAll = () => setSelectedIds(new Set())

  /* ── Bulk actions ── */
  const bulkSetClass = async (classId: string | null) => {
    await Promise.all(
      Array.from(selectedIds).map((imageId) => updateImage.mutateAsync({ imageId, classId, split: undefined })),
    )
    deselectAll()
  }

  const bulkSetSplit = async (split: DatasetSplitValue) => {
    await Promise.all(
      Array.from(selectedIds).map((imageId) => updateImage.mutateAsync({ imageId, classId: undefined, split })),
    )
    deselectAll()
  }

  const bulkDelete = () => {
    modals.openConfirmModal({
      title: 'Delete selected images',
      centered: true,
      children: (
        <Text size="sm">
          Are you sure you want to delete{' '}
          <Text component="span" fw={700}>
            {selectedIds.size}
          </Text>{' '}
          image{selectedIds.size > 1 ? 's' : ''}? This cannot be undone.
        </Text>
      ),
      labels: { confirm: 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        await Promise.all(Array.from(selectedIds).map((id) => deleteImageById.mutateAsync(id)))
        setSelectedIds(new Set())
      },
    })
  }

  const handleAutoSplit = () => {
    const total = splitTrain + splitVal + splitTest
    if (total !== 100) {
      notifications.show({ title: 'Invalid split', message: 'Percentages must sum to 100', color: 'red' })
      return
    }
    autoSplit.mutate({ train: splitTrain, validation: splitVal, test: splitTest })
  }

  /* ── Reset page when filters change ── */
  const handleFilterChange = <T,>(setter: (v: T | null) => void) => {
    return (v: T | null) => {
      setter(v)
      setPage(1)
      deselectAll()
    }
  }

  const handlePerPageChange = (v: string | null) => {
    if (v) {
      setPerPage(Number(v))
      setPage(1)
      deselectAll()
    }
  }

  /* ── Class lookup helper ── */
  const classMap = useMemo(() => {
    const map = new Map<string, { name: string; color: string }>()
    for (const c of classes) map.set(c.id, c)
    return map
  }, [classes])

  return (
    <Box pos="relative">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={2}>Dataset</Title>
            <Text size="sm" c="dimmed" mt={4}>
              {activeProject ? `Manage images for "${activeProject.name}"` : 'Manage dataset images'}
            </Text>
          </div>
          <Group gap="sm" align="center">
            {splitStats.total > 0 && (
              <>
                <Badge variant="light" color="green" size="sm">
                  {splitStats.train} train
                </Badge>
                <Badge variant="light" color="blue" size="sm">
                  {splitStats.validation} val
                </Badge>
                <Badge variant="light" color="red" size="sm">
                  {splitStats.test} test
                </Badge>
                <Badge variant="light" color="gray" size="sm">
                  {splitStats.unassigned} unassigned
                </Badge>
              </>
            )}
            {/* Auto-split dropdown */}
            <Popover width={320} position="bottom-end" shadow="md" withArrow>
              <Popover.Target>
                <Button
                  size="xs"
                  variant="light"
                  leftSection={<MagicWandIcon size={14} />}
                  disabled={splitStats.total === 0}
                >
                  Auto Split
                </Button>
              </Popover.Target>
              <Popover.Dropdown>
                <Stack gap="sm">
                  <Title order={6}>Auto Split</Title>
                  <Group justify="space-between">
                    <Text size="xs">Train</Text>
                    <Badge size="xs" color="green" variant="light">
                      {splitTrain}%
                    </Badge>
                  </Group>
                  <Slider
                    size="xs"
                    color="green"
                    min={0}
                    max={100}
                    value={splitTrain}
                    onChange={(v) => {
                      setSplitTrain(v)
                      const rem = 100 - v
                      const ratio = splitVal + splitTest > 0 ? splitVal / (splitVal + splitTest) : 0.5
                      setSplitVal(Math.round(rem * ratio))
                      setSplitTest(rem - Math.round(rem * ratio))
                    }}
                  />
                  <Group justify="space-between">
                    <Text size="xs">Validation</Text>
                    <Badge size="xs" color="blue" variant="light">
                      {splitVal}%
                    </Badge>
                  </Group>
                  <Slider
                    size="xs"
                    color="blue"
                    min={0}
                    max={100}
                    value={splitVal}
                    onChange={(v) => {
                      setSplitVal(v)
                      setSplitTest(Math.max(0, 100 - splitTrain - v))
                    }}
                  />
                  <Group justify="space-between">
                    <Text size="xs">Test</Text>
                    <Badge size="xs" color="red" variant="light">
                      {splitTest}%
                    </Badge>
                  </Group>
                  <Slider
                    size="xs"
                    color="red"
                    min={0}
                    max={100}
                    value={splitTest}
                    onChange={(v) => {
                      setSplitTest(v)
                      setSplitVal(Math.max(0, 100 - splitTrain - v))
                    }}
                  />
                  {splitTrain + splitVal + splitTest !== 100 && (
                    <Text size="xs" c="red">
                      Must sum to 100% (currently {splitTrain + splitVal + splitTest}%)
                    </Text>
                  )}
                  <Button
                    size="xs"
                    fullWidth
                    leftSection={<MagicWandIcon size={14} />}
                    onClick={handleAutoSplit}
                    loading={autoSplit.isPending}
                    disabled={splitStats.total === 0 || splitTrain + splitVal + splitTest !== 100}
                  >
                    Apply to all images
                  </Button>
                </Stack>
              </Popover.Dropdown>
            </Popover>
          </Group>
        </Group>

        {/* ── Top toolbar: Filters, Sort & View ── */}
        <Card withBorder padding="sm" radius="md">
          <Group gap="md" wrap="wrap" justify="space-between">
            {/* Filters group */}
            <Group gap="xs" align="center">
              <FunnelSimpleIcon size={16} />
              <TextInput
                placeholder="Search filename..."
                size="xs"
                w={200}
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.currentTarget.value)
                  setPage(1)
                }}
              />
              <Select
                placeholder="Class"
                clearable
                size="xs"
                w={150}
                data={classes.map((c) => ({ value: c.id, label: c.name }))}
                value={filterClass}
                onChange={handleFilterChange(setFilterClass)}
              />
              <Select
                placeholder="Split"
                clearable
                size="xs"
                w={140}
                data={SPLIT_FILTER_OPTIONS}
                value={filterSplit}
                onChange={handleFilterChange(setFilterSplit)}
              />
            </Group>

            {/* Sort & View */}
            <Group gap="xs" align="center">
              <SortAscendingIcon size={16} />
              <Select
                size="xs"
                w={140}
                data={SORT_OPTIONS}
                value={sortMode}
                onChange={(v) => {
                  setSortMode((v as SortMode) ?? 'none')
                  setPage(1)
                }}
                allowDeselect={false}
              />
              <SegmentedControl
                size="md"
                data={[
                  {
                    value: 'grid',
                    label: (
                      <Center style={{ gap: 4 }}>
                        <GridFourIcon size={14} />
                        <Text size="xs">Grid</Text>
                      </Center>
                    ),
                  },
                  {
                    value: 'list',
                    label: (
                      <Center style={{ gap: 4 }}>
                        <ListIcon size={14} />
                        <Text size="xs">List</Text>
                      </Center>
                    ),
                  },
                ]}
                value={viewMode}
                onChange={(v) => setViewMode(v as ViewMode)}
              />
            </Group>
          </Group>
        </Card>

        {/* ── Content area ── */}
        <Stack gap="md">
          {/* Toolbar row */}
          {totalImages > 0 && (
            <Group justify="space-between">
              <Group gap="sm">
                <Button size="xs" variant="subtle" onClick={selectAll}>
                  Select all ({images.length})
                </Button>
                {selectedIds.size > 0 && (
                  <Text size="xs" c="dimmed">
                    {selectedIds.size} selected
                  </Text>
                )}
              </Group>
              <Group gap="sm">
                <Text size="xs" c="dimmed">
                  {totalImages} total image{totalImages !== 1 ? 's' : ''}
                </Text>
                <Select
                  size="xs"
                  w={70}
                  data={PER_PAGE_OPTIONS}
                  value={String(perPage)}
                  onChange={handlePerPageChange}
                  allowDeselect={false}
                  comboboxProps={{ position: 'bottom-end' }}
                />
                <Text size="xs" c="dimmed">
                  per page
                </Text>
              </Group>
            </Group>
          )}

          {/* Content */}
          {isLoading ? (
            viewMode === 'grid' ? (
              <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 6 }} spacing="sm">
                {Array.from({ length: Math.min(perPage, 12) }).map((_, i) => (
                  <Skeleton key={i} style={{ aspectRatio: '4/3' }} radius="sm" />
                ))}
              </SimpleGrid>
            ) : (
              <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
                {Array.from({ length: Math.min(perPage, 6) }).map((_, i) => (
                  <Skeleton key={i} height={96} radius="sm" />
                ))}
              </SimpleGrid>
            )
          ) : images.length === 0 ? (
            <Card withBorder p="xl" radius="md" ta="center">
              <Stack align="center" gap="md">
                <ThemeIcon size={56} variant="light" color="gray" radius="xl">
                  <ImagesIcon size={30} weight="thin" />
                </ThemeIcon>
                <Title order={5}>
                  {totalImages === 0 && !debouncedSearch && filterClass === null && filterSplit === null
                    ? 'No images yet'
                    : 'No images match filters'}
                </Title>
                <Text size="sm" c="dimmed">
                  {totalImages === 0 && !debouncedSearch && filterClass === null && filterSplit === null
                    ? 'Go to the Upload page to add images to this project.'
                    : 'Try adjusting the filters above.'}
                </Text>
              </Stack>
            </Card>
          ) : viewMode === 'grid' ? (
            <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 6 }} spacing="sm">
              {images.map((img) => (
                <GridCard
                  key={img.id}
                  image={img}
                  selected={selectedIds.has(img.id)}
                  onSelect={toggleSelect}
                  cls={img.classId ? classMap.get(img.classId) : undefined}
                />
              ))}
            </SimpleGrid>
          ) : (
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
              {images.map((img) => (
                <ListRow
                  key={img.id}
                  image={img}
                  selected={selectedIds.has(img.id)}
                  onSelect={toggleSelect}
                  cls={img.classId ? classMap.get(img.classId) : undefined}
                />
              ))}
            </SimpleGrid>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <Group justify="center" mt="md">
              <Pagination
                total={totalPages}
                value={page}
                onChange={(p) => {
                  setPage(p)
                  deselectAll()
                }}
                size="sm"
              />
            </Group>
          )}

          {/* Spacer when selection toolbar is visible */}
          {selectedIds.size > 0 && <Box h={60} />}
        </Stack>
      </Stack>

      {/* ── Selection toolbar (sticky bottom) ── */}
      {selectedIds.size > 0 && (
        <SelectionToolbar
          count={selectedIds.size}
          classes={classes}
          splitStats={splitStats}
          onAssignClass={bulkSetClass}
          onChangeSplit={bulkSetSplit}
          onDelete={bulkDelete}
          onDeselect={deselectAll}
        />
      )}
    </Box>
  )
}

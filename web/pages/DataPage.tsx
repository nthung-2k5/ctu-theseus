/**
 * Data page – the pool browser + uploader. Every dataset item enters the
 * project here: file drop for vision/audio tasks, inline text entry for
 * text tasks, pasted JSON rows for tabular tasks. New items land in the
 * mutable draft version, ready to be split-assigned and later snapshotted
 * from the Dataset page.
 */

import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Group,
  Pagination,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  ThemeIcon,
  Title,
  Tooltip,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { CloudArrowUpIcon, DatabaseIcon, TrashIcon, UploadSimpleIcon } from '@phosphor-icons/react'
import { AudioLabelingList } from '@public/components/dataset/AudioLabelingList'
import { LabelingProgress } from '@public/components/dataset/LabelingProgress'
import { TabularCsvImporter } from '@public/components/dataset/TabularCsvImporter'
import { TextLabelingList } from '@public/components/dataset/TextLabelingList'
import { VisionLabelingGrid } from '@public/components/dataset/VisionLabelingGrid'
import { DataTable, type DataTableColumn, PageHeader, StatusBadge } from '@public/components/ui'
import { rest, useEden } from '@public/lib/api'
import { SPLIT_COLORS } from '@public/lib/constants'
import { projectDetailQueryOptions, useLabelClasses, useProjectItems } from '@public/lib/queries'
import type { Annotation } from '@public/store/types'
import { getTaskDescriptor } from '@server/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/data')

const SPLIT_OPTIONS = [
  { value: 'train', label: 'Train' },
  { value: 'validation', label: 'Validation' },
  { value: 'test', label: 'Test' },
]

/* ── File upload panel (vision/audio tasks) ── */
function FileUploadPanel({ projectId, accept }: { projectId: string; accept?: string[] }) {
  const eden = useEden()
  const queryClient = useQueryClient()
  const [split, setSplit] = useState<string>('train')
  const [pending, setPending] = useState(false)

  const handleDrop = async (files: File[]) => {
    setPending(true)
    try {
      const { error } = await rest.projects({ projectId }).upload.post({ split: split as 'train', files })
      if (error) throw error
      notifications.show({
        title: 'Upload complete',
        message: `${files.length} file(s) added to the pool`,
        color: 'green',
      })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).items.get.queryKey() })
    } catch {
      notifications.show({ title: 'Upload failed', message: 'One or more files could not be uploaded', color: 'red' })
    } finally {
      setPending(false)
    }
  }

  return (
    <Stack gap="sm">
      <Select
        label="Split"
        data={SPLIT_OPTIONS}
        value={split}
        onChange={(v) => setSplit(v ?? 'train')}
        allowDeselect={false}
      />
      <Dropzone onDrop={handleDrop} accept={accept} loading={pending} radius="md">
        <Group justify="center" gap="md" py="xl" style={{ pointerEvents: 'none' }}>
          <ThemeIcon size={44} variant="light" color="primary" radius="xl">
            <CloudArrowUpIcon size={24} />
          </ThemeIcon>
          <div>
            <Text size="sm" fw={600}>
              Drop files here or click to browse
            </Text>
            <Text size="xs" c="dimmed">
              Added to the <strong>{split}</strong> split of the draft
            </Text>
          </div>
        </Group>
      </Dropzone>
    </Stack>
  )
}

/* ── Inline text entry panel (text tasks) ── */
function TextEntryPanel({ projectId }: { projectId: string }) {
  const form = useForm({ initialValues: { split: 'train', text: '' } })
  const eden = useEden()
  const queryClient = useQueryClient()

  const addItem = useMutation({
    ...eden.api.projects({ projectId }).items.post.mutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).items.get.queryKey() })
      notifications.show({ title: 'Added', message: 'Text item added to the pool', color: 'green' })
      form.setFieldValue('text', '')
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to add item', color: 'red' })
    },
  })

  const handleSubmit = (v: typeof form.values) =>
    addItem.mutate({ items: [{ split: v.split as 'train', textFeatures: { rawText: v.text } }] })

  return (
    <form onSubmit={form.onSubmit(handleSubmit)}>
      <Stack gap="sm">
        <Select label="Split" data={SPLIT_OPTIONS} allowDeselect={false} {...form.getInputProps('split')} />
        <Textarea
          label="Text"
          placeholder="Paste or type the text content for this item"
          autosize
          minRows={4}
          {...form.getInputProps('text')}
        />
        <Group justify="flex-end">
          <Button type="submit" loading={addItem.isPending} disabled={!form.values.text.trim()}>
            Add item
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

/* ── Pool table ── */
type PoolItem = {
  id: string
  externalId: string | null
  splitType: string
  createdAt: string | Date
  annotations?: Annotation[]
}

function PoolTable({
  projectId,
  items,
  isLoading,
  total,
  perPage,
  page,
  onPageChange,
  classNameById,
}: {
  projectId: string
  items: PoolItem[]
  isLoading: boolean
  total: number
  perPage: number
  page: number
  onPageChange: (page: number) => void
  classNameById: Map<string, string>
}) {
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  const eden = useEden()
  const queryClient = useQueryClient()

  // itemId varies per row, so this stays a single reusable mutation (called
  // as `deleteItem.mutate(item.id)`) instead of one useMutation per row,
  // which would call hooks a variable number of times inside .map().
  const deleteItem = useMutation({
    mutationFn: async (itemId: string) => {
      const { data, error } = await rest.items({ itemId }).delete()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).items.get.queryKey() })
    },
    onError: () => {
      notifications.show({
        title: 'Cannot delete',
        message: 'This item may be referenced by a snapshot.',
        color: 'red',
      })
    },
  })

  const columns: DataTableColumn<PoolItem>[] = [
    {
      key: 'externalId',
      header: 'External ID',
      render: (item) => <Text size="xs">{item.externalId ?? item.id.slice(0, 8)}</Text>,
    },
    {
      key: 'split',
      header: 'Split',
      render: (item) => <StatusBadge value={item.splitType} colorMap={SPLIT_COLORS} size="xs" />,
    },
    {
      key: 'label',
      header: 'Label',
      render: (item) => {
        const classId = item.annotations?.find((a) => a.classId)?.classId
        return (
          <Text size="xs" c="dimmed">
            {classId ? classNameById.get(classId) || '—' : '—'}
          </Text>
        )
      },
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
    {
      key: 'actions',
      header: '',
      render: (item) => (
        <Tooltip label="Remove from pool">
          <ActionIcon
            variant="subtle"
            color="red"
            size="sm"
            loading={deleteItem.isPending}
            onClick={() => deleteItem.mutate(item.id)}
          >
            <TrashIcon size={14} />
          </ActionIcon>
        </Tooltip>
      ),
    },
  ]

  return (
    <Stack gap="md">
      <DataTable
        columns={columns}
        data={items}
        getRowKey={(item) => item.id}
        loading={isLoading}
        emptyMessage="No items in the draft pool yet — add some above."
      />

      {totalPages > 1 && (
        <Group justify="center">
          <Pagination total={totalPages} value={page} onChange={onPageChange} size="sm" />
        </Group>
      )}
    </Stack>
  )
}

/* ── Main Data page ── */
export function DataPage() {
  const { projectId } = routeApi.useParams()
  const { page } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(activeProject.task)
  // Every stable task's snapshot has a `label` column (see
  // server/lib/tasks/registry.ts) — classification tasks fill it from a
  // class pick, tabular_regression from a numeric target. Either way, an
  // item without one of these annotations snapshots with a null label.
  const needsAnnotations = descriptor.columns.some((c) => c.kind === 'label')
  const showLabelingSurface = descriptor.annotation.requiresLabelClasses && descriptor.itemSpec.payload !== 'record'

  const perPage = 20
  const { data, isLoading } = useProjectItems(projectId, { page, perPage })
  const { data: classesData } = useLabelClasses(descriptor.annotation.requiresLabelClasses ? projectId : undefined)

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const labeledCount = data?.labeledCount ?? 0
  const classes = classesData?.classes ?? []
  const classNameById = new Map(classes.map((c) => [c.classId, c.name]))

  return (
    <Box>
      <Stack gap="xl">
        <PageHeader
          title="Data"
          description="Add items to the pool and assign them to a split. Snapshot the draft from the Dataset page when ready."
        />

        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
          <Card withBorder p="lg" radius="md">
            <Group gap="sm" mb="md">
              <ThemeIcon size="md" variant="light" color="primary">
                <UploadSimpleIcon size={18} />
              </ThemeIcon>
              <Title order={5}>Add items</Title>
            </Group>
            {descriptor.itemSpec.payload === 'file' && (
              <FileUploadPanel projectId={projectId} accept={descriptor.itemSpec.accept} />
            )}
            {descriptor.itemSpec.payload === 'inline_text' && <TextEntryPanel projectId={projectId} />}
            {descriptor.itemSpec.payload === 'record' && (
              <TabularCsvImporter
                projectId={projectId}
                requiresLabelClasses={descriptor.annotation.requiresLabelClasses}
                classes={classes}
              />
            )}
          </Card>

          <Card withBorder p="lg" radius="md">
            <Group gap="sm" mb="md">
              <ThemeIcon size="md" variant="light" color="teal">
                <DatabaseIcon size={18} />
              </ThemeIcon>
              <Title order={5}>Task</Title>
            </Group>
            <Stack gap="xs">
              <Group justify="space-between">
                <Text size="sm" c="dimmed">
                  Task
                </Text>
                <Text size="sm" fw={500}>
                  {descriptor.label}
                </Text>
              </Group>
              <Group justify="space-between">
                <Text size="sm" c="dimmed">
                  Modality
                </Text>
                <Badge variant="light" size="sm" tt="capitalize">
                  {descriptor.modality}
                </Badge>
              </Group>
              {needsAnnotations && (
                <Group justify="space-between" align="center">
                  <Text size="sm" c="dimmed">
                    Labeling progress
                  </Text>
                  <LabelingProgress labeled={labeledCount} total={total} />
                </Group>
              )}
            </Stack>
          </Card>
        </SimpleGrid>

        {showLabelingSurface && (
          <div>
            <Title order={5} mb="md">
              Label items
            </Title>
            {descriptor.modality === 'vision' && (
              <VisionLabelingGrid projectId={projectId} items={items} classes={classes} />
            )}
            {descriptor.modality === 'text' && (
              <TextLabelingList projectId={projectId} items={items} classes={classes} isLoading={isLoading} />
            )}
            {descriptor.modality === 'audio' && (
              <AudioLabelingList projectId={projectId} items={items} classes={classes} isLoading={isLoading} />
            )}
          </div>
        )}

        <div>
          <Title order={5} mb="md">
            Draft pool
          </Title>
          <PoolTable
            projectId={projectId}
            items={items}
            isLoading={isLoading}
            total={total}
            perPage={perPage}
            page={page}
            onPageChange={(page) => navigate({ search: (prev) => ({ ...prev, page }) })}
            classNameById={classNameById}
          />
        </div>
      </Stack>
    </Box>
  )
}

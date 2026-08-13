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
  JsonInput,
  Loader,
  Pagination,
  Select,
  SimpleGrid,
  Stack,
  Table,
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
import { api } from '@public/lib/api'
import { SPLIT_COLORS } from '@public/lib/constants'
import { useEdenMutation } from '@public/lib/eden-query'
import { queries } from '@public/queries'
import { useLabelClasses } from '@public/queries/classes'
import { useProjectItems } from '@public/queries/dataset'
import type { Annotation, LabelClass } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { getTaskDescriptor } from '@server/lib/tasks'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useParams } from 'wouter'

const SPLIT_OPTIONS = [
  { value: 'train', label: 'Train' },
  { value: 'validation', label: 'Validation' },
  { value: 'test', label: 'Test' },
]

/* ── File upload panel (vision/audio tasks) ── */
function FileUploadPanel({ projectId, accept }: { projectId: string; accept?: string[] }) {
  const queryClient = useQueryClient()
  const [split, setSplit] = useState<string>('train')
  const [pending, setPending] = useState(false)

  const handleDrop = async (files: File[]) => {
    setPending(true)
    try {
      const { error } = await api.projects({ projectId }).upload.post({ split: split as 'train', files })
      if (error) throw error
      notifications.show({
        title: 'Upload complete',
        message: `${files.length} file(s) added to the pool`,
        color: 'green',
      })
      queryClient.invalidateQueries({ queryKey: queries.projects.detail(projectId).queryKey })
      queryClient.invalidateQueries({ queryKey: ['items', projectId] })
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

  const addItem = useEdenMutation(
    (body: { split: string; text: string }) =>
      api.projects({ projectId }).items.post({
        items: [{ split: body.split as 'train', textFeatures: { rawText: body.text } }],
      }),
    [queries.projects.detail(projectId).queryKey],
    {
      onSuccess: () => {
        notifications.show({ title: 'Added', message: 'Text item added to the pool', color: 'green' })
        form.setFieldValue('text', '')
      },
      onError: () => {
        notifications.show({ title: 'Error', message: 'Failed to add item', color: 'red' })
      },
    },
  )

  return (
    <form onSubmit={form.onSubmit((v) => addItem.mutate(v))}>
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

/* ── Tabular record entry panel (record tasks) ──
 * NOTE: this only writes tabularFeatures.featuresJson — it doesn't attach a
 * label/target annotation, so tabular_classification and tabular_regression
 * snapshots built from rows added here will have a null label column. A
 * proper tabular importer (CSV upload with a target-column picker) is
 * follow-up work; this covers unlabeled feature entry only. */
function RecordEntryPanel({ projectId }: { projectId: string }) {
  const form = useForm({ initialValues: { split: 'train', rows: '' } })

  const addItems = useEdenMutation(
    (body: { split: string; records: Record<string, unknown>[] }) =>
      api.projects({ projectId }).items.post({
        items: body.records.map((featuresJson) => ({
          split: body.split as 'train',
          tabularFeatures: { featuresJson },
        })),
      }),
    [queries.projects.detail(projectId).queryKey],
    {
      onSuccess: (data) => {
        notifications.show({
          title: 'Added',
          message: `${data.results.length} row(s) added to the pool`,
          color: 'green',
        })
        form.setFieldValue('rows', '')
      },
      onError: () => {
        notifications.show({ title: 'Error', message: 'Failed to add rows — check the JSON', color: 'red' })
      },
    },
  )

  const handleSubmit = (values: typeof form.values) => {
    let records: unknown
    try {
      records = JSON.parse(values.rows)
    } catch {
      form.setFieldError('rows', 'Invalid JSON')
      return
    }
    if (!Array.isArray(records) || records.length === 0) {
      form.setFieldError('rows', 'Expected a non-empty JSON array of row objects')
      return
    }
    addItems.mutate({ split: values.split, records: records as Record<string, unknown>[] })
  }

  return (
    <form onSubmit={form.onSubmit(handleSubmit)}>
      <Stack gap="sm">
        <Select label="Split" data={SPLIT_OPTIONS} allowDeselect={false} {...form.getInputProps('split')} />
        <JsonInput
          label="Rows"
          description='A JSON array of row objects, e.g. [{"age": 34, "income": 52000}]'
          placeholder='[{"col1": 1, "col2": "a"}]'
          autosize
          minRows={6}
          maxRows={16}
          formatOnBlur
          {...form.getInputProps('rows')}
        />
        <Group justify="flex-end">
          <Button type="submit" loading={addItems.isPending}>
            Add rows
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

/* ── Pool table ── */
function PoolTable({ projectId, requiresLabelClasses }: { projectId: string; requiresLabelClasses: boolean }) {
  const [page, setPage] = useState(1)
  const perPage = 20
  const { data, isLoading } = useProjectItems(projectId, { page, perPage })
  const { data: classesData } = useLabelClasses(requiresLabelClasses ? projectId : undefined)
  const classNameById = new Map<string, string>(
    (classesData?.classes ?? []).map((c: LabelClass) => [c.classId, c.name]),
  )

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  const deleteItem = useEdenMutation(
    (itemId: string) => api.items({ itemId }).delete(),
    [queries.projects.detail(projectId).queryKey, ['items', projectId]],
    {
      onError: () => {
        notifications.show({
          title: 'Cannot delete',
          message: 'This item may be referenced by a snapshot.',
          color: 'red',
        })
      },
    },
  )

  if (isLoading) {
    return (
      <Card withBorder p="xl" radius="md" ta="center">
        <Loader size="sm" />
      </Card>
    )
  }

  if (items.length === 0) {
    return (
      <Card withBorder p="xl" radius="md" ta="center">
        <Stack align="center" gap="sm">
          <ThemeIcon size={48} variant="light" color="gray" radius="xl">
            <DatabaseIcon size={26} weight="thin" />
          </ThemeIcon>
          <Text size="sm" c="dimmed">
            No items in the draft pool yet — add some above.
          </Text>
        </Stack>
      </Card>
    )
  }

  return (
    <Stack gap="md">
      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>External ID</Table.Th>
            <Table.Th>Split</Table.Th>
            <Table.Th>Label</Table.Th>
            <Table.Th>Created</Table.Th>
            <Table.Th />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {items.map(
            (item: {
              id: string
              externalId: string | null
              splitType: string
              createdAt: string | Date
              annotations?: Annotation[]
            }) => {
              const classId = item.annotations?.find((a) => a.classId)?.classId
              return (
                <Table.Tr key={item.id}>
                  <Table.Td>
                    <Text size="xs">{item.externalId ?? item.id.slice(0, 8)}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Badge size="xs" variant="light" color={SPLIT_COLORS[item.splitType] ?? 'gray'} tt="capitalize">
                      {item.splitType}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {classId ? classNameById.get(classId) || '—' : '—'}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {new Date(item.createdAt).toLocaleDateString()}
                    </Text>
                  </Table.Td>
                  <Table.Td>
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
                  </Table.Td>
                </Table.Tr>
              )
            },
          )}
        </Table.Tbody>
      </Table>

      {totalPages > 1 && (
        <Group justify="center">
          <Pagination total={totalPages} value={page} onChange={setPage} size="sm" />
        </Group>
      )}
    </Stack>
  )
}

/* ── Main Data page ── */
export function DataPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id
  const activeProject = useProjectStore((s) => s.activeProject)

  if (!activeProject) {
    return (
      <Card withBorder p="xl" radius="md" ta="center">
        <Loader size="sm" />
      </Card>
    )
  }

  const descriptor = getTaskDescriptor(activeProject.task)

  return (
    <Box>
      <Stack gap="xl">
        <div>
          <Title order={2}>Data</Title>
          <Text size="sm" c="dimmed" mt={4}>
            Add items to the pool and assign them to a split. Snapshot the draft from the Dataset page when ready.
          </Text>
        </div>

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
            {descriptor.itemSpec.payload === 'record' && <RecordEntryPanel projectId={projectId} />}
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
              <Group justify="space-between">
                <Text size="sm" c="dimmed">
                  Requires label classes
                </Text>
                <Text size="sm">{descriptor.annotation.requiresLabelClasses ? 'Yes' : 'No'}</Text>
              </Group>
            </Stack>
          </Card>
        </SimpleGrid>

        <div>
          <Title order={5} mb="md">
            Draft pool
          </Title>
          <PoolTable projectId={projectId} requiresLabelClasses={descriptor.annotation.requiresLabelClasses} />
        </div>
      </Stack>
    </Box>
  )
}

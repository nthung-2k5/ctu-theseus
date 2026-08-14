import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  ColorInput,
  Group,
  Modal,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { PencilSimpleIcon, PlusIcon, TagIcon, TrashIcon } from '@phosphor-icons/react'
import { confirmDelete, EmptyState, PageHeader } from '@public/components/ui'
import { useEden } from '@public/lib/api'
import { projectDetailQueryOptions, useLabelClasses } from '@public/lib/queries'
import type { LabelClass } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/classes')

/* ── Create / Edit Class Modal ── */
function ClassFormModal({
  projectId,
  existing,
  onClose,
}: {
  projectId: string
  existing?: LabelClass
  onClose: () => void
}) {
  const isEdit = !!existing

  const form = useForm({
    initialValues: {
      name: existing?.name ?? '',
      description: existing?.description ?? '',
      uiColorHex: existing?.uiColorHex ?? '#e03131',
    },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Class name is required'),
      uiColorHex: (v) => (/^#[0-9a-fA-F]{6}$/.test(v) ? null : 'Invalid hex color'),
    },
  })

  const eden = useEden()
  const queryClient = useQueryClient()
  const classesQueryKey = eden.api.projects({ projectId }).classes.get.queryKey()

  const createClass = useMutation({
    ...eden.api.projects({ projectId }).classes.post.mutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: classesQueryKey })
      notifications.show({ title: 'Class created', message: `"${form.values.name}" added`, color: 'green' })
      form.reset()
      onClose()
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to create class', color: 'red' })
    },
  })

  const updateClass = useMutation({
    ...eden.api
      .projects({ projectId })
      .classes({ classId: existing?.classId ?? '' })
      .patch.mutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: classesQueryKey })
      notifications.show({ title: 'Class updated', message: `"${form.values.name}" updated`, color: 'green' })
      onClose()
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to update class', color: 'red' })
    },
  })

  const handleSubmit = (values: typeof form.values) => {
    const body = {
      name: values.name.trim(),
      description: values.description.trim() || undefined,
      uiColorHex: values.uiColorHex,
    }
    if (isEdit) {
      updateClass.mutate(body)
    } else {
      createClass.mutate(body)
    }
  }

  return (
    <form onSubmit={form.onSubmit(handleSubmit)}>
      <Stack gap="md">
        <TextInput label="Class name" placeholder="e.g. pedestrian, cat, positive" {...form.getInputProps('name')} />
        <Textarea
          label="Description"
          placeholder="Describe what this class represents for annotators"
          autosize
          minRows={2}
          {...form.getInputProps('description')}
        />
        <ColorInput
          label="Color"
          description="Used for bounding boxes and UI indicators"
          format="hex"
          swatches={[
            '#e03131',
            '#2f9e44',
            '#1971c2',
            '#f08c00',
            '#9c36b5',
            '#0c8599',
            '#e8590c',
            '#6741d9',
            '#3bc9db',
            '#ff6b6b',
            '#51cf66',
            '#339af0',
            '#fcc419',
            '#cc5de8',
            '#20c997',
          ]}
          {...form.getInputProps('uiColorHex')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" loading={createClass.isPending || updateClass.isPending}>
            {isEdit ? 'Save Changes' : 'Create Class'}
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

/* ── Class Card ── */
function ClassCard({
  cls,
  projectId,
  onEdit,
}: {
  cls: LabelClass
  projectId: string
  onEdit: (cls: LabelClass) => void
}) {
  const eden = useEden()
  const queryClient = useQueryClient()

  const deleteClass = useMutation({
    ...eden.api.projects({ projectId }).classes({ classId: cls.classId }).delete.mutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).classes.get.queryKey() })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      notifications.show({ title: 'Deleted', message: `"${cls.name}" removed`, color: 'green' })
    },
    onError: () => {
      notifications.show({
        title: 'Cannot delete',
        message: 'This class is referenced by existing annotations. Remove them first.',
        color: 'red',
      })
    },
  })

  const handleDelete = () => {
    confirmDelete({
      title: 'Delete class',
      message: (
        <>
          Are you sure you want to delete <strong>{cls.name}</strong>? This will fail if there are annotations using
          this class.
        </>
      ),
      onConfirm: () => deleteClass.mutate(),
    })
  }

  return (
    <Card withBorder p="lg" radius="md" className="card-elevated">
      <Group justify="space-between" mb="sm">
        <Group gap="sm">
          <Box
            w={20}
            h={20}
            style={{
              borderRadius: 4,
              backgroundColor: cls.uiColorHex ?? 'var(--mantine-color-gray-6)',
              border: '1px solid var(--mantine-color-dark-4)',
            }}
          />
          <Title order={5}>{cls.name}</Title>
        </Group>
        <Group gap={4}>
          <Tooltip label="Edit">
            <ActionIcon variant="subtle" color="gray" onClick={() => onEdit(cls)}>
              <PencilSimpleIcon size={16} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Delete">
            <ActionIcon variant="subtle" color="red" onClick={handleDelete} loading={deleteClass.isPending}>
              <TrashIcon size={16} />
            </ActionIcon>
          </Tooltip>
        </Group>
      </Group>

      {cls.description ? (
        <Text size="sm" c="dimmed" lineClamp={2}>
          {cls.description}
        </Text>
      ) : (
        <Text size="sm" c="dimmed" fs="italic">
          No description
        </Text>
      )}

      <Text size="xs" c="dimmed" mt="sm">
        Created {cls.createdAt ? new Date(cls.createdAt).toLocaleDateString() : '—'}
      </Text>
    </Card>
  )
}

/* ── Main Classes Page ── */
export function ClassesPage() {
  const { projectId } = routeApi.useParams()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const { data, isLoading } = useLabelClasses(projectId)
  const classes = data?.classes ?? []

  const [createOpened, { open: openCreate, close: closeCreate }] = useDisclosure(false)
  const [editingClass, setEditingClass] = useState<LabelClass | null>(null)

  return (
    <Box>
      <Stack gap="xl">
        <PageHeader
          title="Label Classes"
          description="Define the classification labels for your dataset. Each annotation references one of these classes."
          actions={
            <Button leftSection={<PlusIcon size={16} />} onClick={openCreate}>
              Add Class
            </Button>
          }
        />

        {/* Classes grid */}
        {isLoading ? (
          <EmptyState loading />
        ) : classes.length === 0 ? (
          <EmptyState
            icon={TagIcon}
            title="No label classes yet"
            description={
              <>
                Add classification labels that annotators can assign to dataset items. This project uses the{' '}
                <strong>{activeProject.task.replace(/_/g, ' ')}</strong> task.
              </>
            }
            action={
              <Button leftSection={<PlusIcon size={14} />} onClick={openCreate}>
                Create First Class
              </Button>
            }
          />
        ) : (
          <>
            <Group gap="sm">
              <Badge variant="light" size="lg">
                {classes.length} class(es)
              </Badge>
            </Group>
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
              {classes.map((cls) => (
                <ClassCard key={cls.classId} cls={cls} projectId={projectId} onEdit={setEditingClass} />
              ))}
            </SimpleGrid>
          </>
        )}
      </Stack>

      {/* Create Modal */}
      <Modal opened={createOpened} onClose={closeCreate} title="Create Label Class" centered>
        <ClassFormModal projectId={projectId} onClose={closeCreate} />
      </Modal>

      {/* Edit Modal */}
      <Modal opened={!!editingClass} onClose={() => setEditingClass(null)} title="Edit Label Class" centered>
        {editingClass && (
          <ClassFormModal projectId={projectId} existing={editingClass} onClose={() => setEditingClass(null)} />
        )}
      </Modal>
    </Box>
  )
}

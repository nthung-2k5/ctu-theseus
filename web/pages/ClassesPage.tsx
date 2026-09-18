/**
 * Classes page – the label-class registry for classification tasks: create,
 * edit, and delete the classes annotators/bulk actions assign to items.
 * Assigning classes to items themselves happens elsewhere (per-item on the
 * Data upload flow's quick-assign, and in bulk on the Dataset page) — this
 * page only manages the class definitions.
 */

import { Box, Button, ColorInput, Group, Modal, Stack, Text, Textarea, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { PencilSimpleIcon, PlusIcon, TagIcon, TrashIcon } from '@phosphor-icons/react'
import {
  confirmDelete,
  DataTable,
  type DataTableColumn,
  EmptyState,
  PageHeader,
  QueryBoundary,
} from '@public/components/ui'
import { useEden } from '@public/lib/api'
import { invalidateProjectScope, projectDetailQueryOptions, useLabelClasses } from '@public/lib/queries'
import type { LabelClass } from '@public/store/types'
import { getTaskDescriptor } from '@server/lib/tasks'
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
  // Project scope: class names also render in the item list, and the sidebar
  // badge counts them off project detail.
  const invalidate = () => invalidateProjectScope(queryClient, projectId)

  const createClass = useMutation({
    ...eden.api.projects({ projectId }).classes.post.mutationOptions(),
    onSuccess: () => {
      invalidate()
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
      invalidate()
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

/* ── Class Row Actions (edit + delete) ── */
function ClassActions({
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
      invalidateProjectScope(queryClient, projectId)
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
    <Group gap="xs" wrap="nowrap">
      <Button
        size="xs"
        variant="subtle"
        color="gray"
        leftSection={<PencilSimpleIcon size={14} />}
        onClick={() => onEdit(cls)}
      >
        Edit
      </Button>
      <Button
        size="xs"
        variant="subtle"
        color="red"
        leftSection={<TrashIcon size={14} />}
        loading={deleteClass.isPending}
        onClick={handleDelete}
      >
        Delete
      </Button>
    </Group>
  )
}

/* ── Main Classes page ── */
export function ClassesPage() {
  const { projectId } = routeApi.useParams()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(activeProject.task)
  const { data, isLoading, isError, refetch } = useLabelClasses(projectId)
  const classes = data?.classes ?? []

  const [createOpened, { open: openCreate, close: closeCreate }] = useDisclosure(false)
  const [editingClass, setEditingClass] = useState<LabelClass | null>(null)

  if (!descriptor.annotation.requiresLabelClasses) {
    return (
      <Box>
        <Stack gap="xl">
          <PageHeader title="Classes" description="The label-class registry for this project's task." />
          <EmptyState
            icon={TagIcon}
            title="Not used by this task"
            description={`The "${descriptor.label}" task doesn't use label classes.`}
          />
        </Stack>
      </Box>
    )
  }

  const classColumns: DataTableColumn<LabelClass>[] = [
    {
      key: 'name',
      header: 'Class',
      render: (cls) => (
        <Group gap="sm" wrap="nowrap">
          <Box
            w={16}
            h={16}
            style={{
              borderRadius: 4,
              backgroundColor: cls.uiColorHex ?? 'var(--mantine-color-gray-6)',
              border: '1px solid var(--mantine-color-dark-4)',
              flexShrink: 0,
            }}
          />
          <Text size="sm" fw={600}>
            {cls.name}
          </Text>
        </Group>
      ),
    },
    {
      key: 'description',
      header: 'Description',
      render: (cls) =>
        cls.description ? (
          <Text size="sm" c="dimmed" lineClamp={2}>
            {cls.description}
          </Text>
        ) : (
          <Text size="sm" c="dimmed" fs="italic">
            No description
          </Text>
        ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (cls) => <ClassActions cls={cls} projectId={projectId} onEdit={setEditingClass} />,
    },
  ]

  return (
    <Box>
      <Stack gap="xl">
        <PageHeader
          title="Classes"
          description="Define the classification labels used to annotate this project's dataset."
          actions={
            <Button leftSection={<PlusIcon size={14} />} onClick={openCreate}>
              Add Class
            </Button>
          }
        />

        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {classes.length === 0 ? (
            <EmptyState
              icon={TagIcon}
              title="No label classes yet"
              description={
                <>
                  Add classification labels that items can be assigned to. This project uses the{' '}
                  <strong>{descriptor.label}</strong> task.
                </>
              }
              action={
                <Button leftSection={<PlusIcon size={14} />} onClick={openCreate}>
                  Create First Class
                </Button>
              }
            />
          ) : (
            <DataTable columns={classColumns} data={classes} getRowKey={(cls) => cls.classId} />
          )}
        </QueryBoundary>

        <Modal opened={createOpened} onClose={closeCreate} title="Create Label Class" centered>
          <ClassFormModal projectId={projectId} onClose={closeCreate} />
        </Modal>

        <Modal opened={!!editingClass} onClose={() => setEditingClass(null)} title="Edit Label Class" centered>
          {editingClass && (
            <ClassFormModal projectId={projectId} existing={editingClass} onClose={() => setEditingClass(null)} />
          )}
        </Modal>
      </Stack>
    </Box>
  )
}

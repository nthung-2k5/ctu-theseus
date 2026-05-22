import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  ColorSwatch,
  Group,
  Skeleton,
  Stack,
  Table,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
} from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { PencilSimpleIcon, PlusIcon, TagIcon, TrashIcon } from '@phosphor-icons/react'
import { CreateClassModal } from '@public/components/classes/CreateClassModal'
import { EditClassModal } from '@public/components/classes/EditClassModal'
import { api } from '@public/lib/api'
import { useEdenMutation, useEdenQuery } from '@public/lib/eden-query'
import type { DatasetClass } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { useQueryClient } from '@tanstack/react-query'
import { useParams } from 'wouter'

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export const ClassPage = () => {
  const params = useParams<{ id: string }>()
  const projectId = params.id
  const activeProject = useProjectStore((s) => s.activeProject)
  const queryClient = useQueryClient()

  const { data, isLoading } = useEdenQuery(
    ['projects', projectId, 'classes'],
    api.projects({ projectId }).classes.get,
  )

  const classes = data?.classes ?? []

  /* ── Delete mutation ── */
  const deleteClass = useEdenMutation(
    (classId: string) => api.classes({ classId }).delete(),
    {
      onSuccess: async () => {
        await queryClient.invalidateQueries({ queryKey: ['projects', projectId] })
        notifications.show({ title: 'Class deleted', message: 'The class has been removed', color: 'green' })
      },
      onError: () => {
        notifications.show({ title: 'Error', message: 'Failed to delete class', color: 'red' })
      },
    },
  )

  /* ── Modal openers ── */
  const openCreateModal = () => {
    modals.open({
      title: 'Add class',
      centered: true,
      children: <CreateClassModal projectId={projectId} existingCount={classes.length} />,
    })
  }

  const openEditModal = (cls: DatasetClass) => {
    modals.open({
      title: 'Edit class',
      centered: true,
      children: <EditClassModal projectId={projectId} cls={cls} />,
    })
  }

  const openDeleteConfirm = (cls: DatasetClass) => {
    modals.openConfirmModal({
      title: 'Delete class',
      centered: true,
      children: (
        <Text size="sm">
          Are you sure you want to delete the class{' '}
          <Text component="span" fw={600}>
            {cls.name}
          </Text>
          ? This will also remove all labels assigned to this class and cannot be undone.
        </Text>
      ),
      labels: { confirm: 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: () => deleteClass.mutate(cls.id),
    })
  }

  /* ── Render ── */
  return (
    <Box>
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between">
          <div>
            <Title order={2}>Classes</Title>
            <Text size="sm" c="dimmed" mt={4}>
              {activeProject
                ? `Define the object classes for "${activeProject.name}"`
                : 'Define the object classes for this project'}
            </Text>
          </div>
          <Button leftSection={<PlusIcon size={18} />} onClick={openCreateModal}>
            Add class
          </Button>
        </Group>

        {/* Content */}
        {isLoading ? (
          <Stack gap="sm">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} height={52} radius="md" />
            ))}
          </Stack>
        ) : classes.length === 0 ? (
          <Card withBorder p="xl" radius="md" ta="center">
            <Stack align="center" gap="md">
              <ThemeIcon size={56} variant="light" color="primary" radius="xl">
                <TagIcon size={30} weight="thin" />
              </ThemeIcon>
              <Title order={4}>No classes yet</Title>
              <Text size="sm" c="dimmed" maw={400}>
                Classes define what objects your model will learn to detect. Add at least one class to get started.
              </Text>
              <Button leftSection={<PlusIcon size={18} />} onClick={openCreateModal}>
                Add first class
              </Button>
            </Stack>
          </Card>
        ) : (
          <Card withBorder radius="md" p={0}>
            <Group px="lg" py="md" justify="space-between">
              <Text size="sm" fw={500} c="dimmed">
                {classes.length} {classes.length === 1 ? 'class' : 'classes'} defined
              </Text>
              <Badge variant="light" color="primary" size="sm">
                {classes.length} total
              </Badge>
            </Group>

            <Table highlightOnHover verticalSpacing="sm" horizontalSpacing="lg">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Color</Table.Th>
                  <Table.Th>Name</Table.Th>
                  <Table.Th style={{ width: 96 }} />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {classes.map((cls) => (
                  <Table.Tr key={cls.id}>
                    <Table.Td>
                      <ColorSwatch color={cls.color} size={22} />
                    </Table.Td>
                    <Table.Td>
                      <Text size="sm" fw={500}>
                        {cls.name}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Group gap={4} justify="flex-end">
                        <Tooltip label="Edit class" withArrow>
                          <ActionIcon variant="subtle" color="gray" onClick={() => openEditModal(cls)}>
                            <PencilSimpleIcon size={16} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Delete class" withArrow>
                          <ActionIcon
                            variant="subtle"
                            color="red"
                            loading={deleteClass.isPending}
                            onClick={() => openDeleteConfirm(cls)}
                          >
                            <TrashIcon size={16} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>
        )}
      </Stack>
    </Box>
  )
}

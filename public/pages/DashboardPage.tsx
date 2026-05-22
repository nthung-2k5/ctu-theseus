import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Group,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { FolderSimpleIcon, PencilSimpleIcon, PlusIcon } from '@phosphor-icons/react'
import { api } from '@public/lib/api'
import { assert } from '@public/lib/assert'
import { useEdenMutation, useEdenQuery } from '@public/lib/eden-query'
import type { Project } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { useLocation } from 'wouter'

const CreateProjectModal = () => {
  const form = useForm({
    initialValues: { name: '', description: '' },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
    },
  })

  const queryClient = useQueryClient()

  const createProject = useEdenMutation(api.projects.post, {
    onSuccess: ({ project }) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      notifications.show({ title: 'Project created', message: `"${project.name}" is ready`, color: 'green' })
      modals.closeAll()
    },
    onError: (error) => {
      notifications.show({
        title: 'Error',
        message: typeof error.value === 'string' ? error.value : (error.value?.message ?? 'Failed to create project'),
        color: 'red',
      })
    },
    onSettled: () => {
      form.reset()
    },
  })

  return (
    <form onSubmit={form.onSubmit((values) => createProject.mutate(values))}>
      <Stack gap="md">
        <TextInput label="Project name" placeholder="e.g. Traffic Signs" {...form.getInputProps('name')} />
        <Textarea
          label="Description"
          placeholder="What are you training the model to detect?"
          autosize
          minRows={3}
          {...form.getInputProps('description')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={modals.closeAll}>
            Cancel
          </Button>
          <Button type="submit">Create</Button>
        </Group>
      </Stack>
    </form>
  )
}

const UpdateProjectModal = ({ project }: { project: Project }) => {
  const form = useForm({
    initialValues: { name: project.name, description: project.description },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
    },
  })

  const queryClient = useQueryClient()

  const updateProject = useEdenMutation(api.projects({ projectId: project.id }).patch, {
    onSuccess: ({ project }) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      notifications.show({ title: 'Project updated', message: `"${project.name}" has been updated`, color: 'green' })
      modals.closeAll()
    },
    onError: (error) => {
      assert(error.status === 404 || error.status === 422)
      notifications.show({
        title: 'Error',
        message: error.status === 404 ? error.value : (error.value?.message ?? 'Failed to update project'),
        color: 'red',
      })
    },
    onSettled: () => {
      form.reset()
    },
  })

  return (
    <form onSubmit={form.onSubmit((values) => updateProject.mutate(values))}>
      <Stack gap="md">
        <TextInput label="Project name" placeholder="e.g. Traffic Signs" {...form.getInputProps('name')} />
        <Textarea
          label="Description"
          placeholder="What are you training the model to detect?"
          autosize
          minRows={3}
          {...form.getInputProps('description')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={modals.closeAll}>
            Cancel
          </Button>
          <Button type="submit">Save Changes</Button>
        </Group>
      </Stack>
    </form>
  )
}

export function DashboardPage() {
  const [, setLocation] = useLocation()

  const { data, isLoading } = useEdenQuery(['projects'], api.projects.get)
  const setActiveProject = useProjectStore((s) => s.setActiveProject)

  const projects = data?.projects ?? []

  useEffect(() => {
    setActiveProject(null)
  }, [setActiveProject])

  const openProject = (project: Project) => {
    setLocation(`/project/${project.id}`)
  }

  const openCreateProjectModal = () => {
    modals.open({
      title: 'Create new project',
      centered: true,
      children: <CreateProjectModal />,
    })
  }

  const openUpdateProjectModal = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>, project: Project) => {
    e.stopPropagation()
    modals.open({
      title: 'Edit project',
      centered: true,
      children: <UpdateProjectModal project={project} />,
    })
  }

  return (
    <Box>
      <Group justify="space-between" mb="xl">
        <div>
          <Title order={2}>Projects</Title>
          <Text size="sm" c="dimmed" mt={4}>
            Manage your image detection projects
          </Text>
        </div>
        <Button leftSection={<PlusIcon size={18} />} onClick={openCreateProjectModal}>
          New project
        </Button>
      </Group>

      {isLoading ? (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} height={160} radius="md" />
          ))}
        </SimpleGrid>
      ) : projects.length === 0 ? (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="md">
            <FolderSimpleIcon size={48} weight="thin" />
            <Title order={4}>No projects yet</Title>
            <Text size="sm" c="dimmed">
              Create your first project to get started with AI image detection.
            </Text>
            <Button leftSection={<PlusIcon size={18} />} onClick={openCreateProjectModal}>
              Create project
            </Button>
          </Stack>
        </Card>
      ) : (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg">
          {projects.map((project) => (
            <Card
              key={project.id}
              withBorder
              padding="lg"
              radius="md"
              className="card-elevated"
              style={{ cursor: 'pointer' }}
              onClick={() => openProject(project)}
            >
              <Stack gap="sm">
                <Group justify="space-between">
                  <Title order={5}>{project.name}</Title>
                  <Group gap="xs">
                    <Badge variant="light" color="primary" size="sm">
                      Active
                    </Badge>
                    <ActionIcon variant="subtle" color="gray" onClick={(e) => openUpdateProjectModal(e, project)}>
                      <PencilSimpleIcon size={16} />
                    </ActionIcon>
                  </Group>
                </Group>
                <Text size="sm" c="dimmed" lineClamp={2}>
                  {project.description || 'No description provided'}
                </Text>
                {/* <Group gap="lg" mt="xs">
                  <Group gap={4}>
                    <ImageIcon size={16} />
                    <Text size="xs" c="dimmed">
                      {project.imageCount} images
                    </Text>
                  </Group>
                  <Group gap={4}>
                    <TagChevronIcon size={16} />
                    <Text size="xs" c="dimmed">
                      {project.classCount} classes
                    </Text>
                  </Group>
                </Group> */}
              </Stack>
            </Card>
          ))}
        </SimpleGrid>
      )}
    </Box>
  )
}

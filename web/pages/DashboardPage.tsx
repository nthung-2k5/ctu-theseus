import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Group,
  Select,
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
import { UpdateProjectModal } from '@public/components/UpdateProjectModal'
import { EmptyState, PageHeader } from '@public/components/ui'
import { useEden } from '@public/lib/api'
import { useProjects } from '@public/lib/queries'
import type { DatasetModality, ProjectTask } from '@server/lib/enums'
import { taskRegistry } from '@server/lib/tasks'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'

const MODALITY_GROUP_LABELS: Record<DatasetModality, string> = {
  text: 'Text',
  vision: 'Vision',
  audio: 'Audio',
  tabular: 'Tabular',
}

const MODALITY_COLORS: Record<DatasetModality, string> = {
  text: 'blue',
  vision: 'green',
  audio: 'orange',
  tabular: 'grape',
}

/**
 * Every task in the registry, grouped by modality — planned (non-Ludwig)
 * tasks are shown disabled with a "coming soon" label instead of hidden, so
 * users can see the roadmap without being able to submit a task the backend
 * would 422 on (see server/routes/projects.ts's descriptor.backend check).
 */
const TASK_OPTIONS = (Object.keys(MODALITY_GROUP_LABELS) as DatasetModality[]).map((modality) => ({
  group: MODALITY_GROUP_LABELS[modality],
  items: Object.values(taskRegistry)
    .filter((d) => d.modality === modality)
    .map((d) => ({
      value: d.id,
      label: d.backend === 'ludwig' ? d.label : `${d.label} (coming soon)`,
      disabled: d.backend !== 'ludwig',
    })),
}))

function taskLabel(task: string): string {
  return taskRegistry[task as ProjectTask]?.label ?? task
}

const CreateProjectModal = () => {
  const form = useForm({
    initialValues: { name: '', description: '', task: '' as string },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
      task: (v) => (v ? null : 'Please select a task'),
    },
  })

  const eden = useEden()
  const queryClient = useQueryClient()

  const createProject = useMutation({
    ...eden.api.projects.post.mutationOptions(),
    onSuccess: ({ project }) => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects.get.queryKey() })
      notifications.show({ title: 'Project created', message: `"${project.name}" is ready`, color: 'green' })
      form.reset()
      modals.closeAll()
    },
    onError: (error) => {
      const value: unknown = error.value
      const message = typeof value === 'string' ? value : (value as { message?: string } | undefined)?.message
      notifications.show({ title: 'Error', message: message ?? 'Failed to create project', color: 'red' })
    },
  })

  return (
    <form
      onSubmit={form.onSubmit((values) =>
        createProject.mutate({ name: values.name, description: values.description, task: values.task as ProjectTask }),
      )}
    >
      <Stack gap="md">
        <TextInput label="Project name" placeholder="e.g. Traffic Signs" {...form.getInputProps('name')} />
        <Textarea
          label="Description"
          placeholder="What is this project about?"
          autosize
          minRows={3}
          {...form.getInputProps('description')}
        />
        <Select
          label="Task"
          placeholder="Select ML task"
          data={TASK_OPTIONS}
          searchable
          {...form.getInputProps('task')}
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

export function DashboardPage() {
  const navigate = useNavigate()

  const { data, isLoading } = useProjects()

  const projects = data?.projects ?? []

  const openProject = (projectId: string) => {
    navigate({ to: '/project/$projectId', params: { projectId } })
  }

  const openCreateProjectModal = () => {
    modals.open({
      title: 'Create new project',
      centered: true,
      children: <CreateProjectModal />,
    })
  }

  const openUpdateProjectModal = (
    e: React.MouseEvent<HTMLButtonElement, MouseEvent>,
    projectId: string,
    projectName: string,
    projectDescription: string | null,
  ) => {
    e.stopPropagation()
    modals.open({
      title: 'Edit project',
      centered: true,
      children: (
        <UpdateProjectModal projectId={projectId} projectName={projectName} projectDescription={projectDescription} />
      ),
    })
  }

  return (
    <Box>
      <PageHeader
        title="Projects"
        description="Manage your machine learning projects"
        actions={
          <Button leftSection={<PlusIcon size={18} />} onClick={openCreateProjectModal}>
            New project
          </Button>
        }
      />

      <Box mt="xl">
        {isLoading ? (
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg">
            {Array.from({ length: 6 }, (_, i) => `skeleton-${i}`).map((key) => (
              <Skeleton key={key} height={160} radius="md" />
            ))}
          </SimpleGrid>
        ) : projects.length === 0 ? (
          <EmptyState
            icon={FolderSimpleIcon}
            title="No projects yet"
            description="Create your first project to get started with AI training."
            action={
              <Button leftSection={<PlusIcon size={18} />} onClick={openCreateProjectModal}>
                Create project
              </Button>
            }
          />
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
                onClick={() => openProject(project.id)}
              >
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Title order={5}>{project.name}</Title>
                    <Group gap="xs">
                      <Badge variant="light" color={MODALITY_COLORS[taskRegistry[project.task].modality]} size="sm">
                        {taskLabel(project.task)}
                      </Badge>
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        onClick={(e) => openUpdateProjectModal(e, project.id, project.name, project.description)}
                      >
                        <PencilSimpleIcon size={16} />
                      </ActionIcon>
                    </Group>
                  </Group>
                  <Text size="sm" c="dimmed" lineClamp={2}>
                    {project.description || 'No description provided'}
                  </Text>
                </Stack>
              </Card>
            ))}
          </SimpleGrid>
        )}
      </Box>
    </Box>
  )
}

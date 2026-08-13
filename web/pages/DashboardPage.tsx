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
import { api } from '@public/lib/api'
import { assert } from '@public/lib/assert'
import { useEdenMutation } from '@public/lib/eden-query'
import { queries } from '@public/queries'
import { useProjects } from '@public/queries/project'
import { useProjectStore } from '@public/store/useProjectStore'
import { useEffect } from 'react'
import { useLocation } from 'wouter'

const TASK_OPTIONS = [
  {
    group: 'Text',
    items: [
      { value: 'text_classification', label: 'Text Classification' },
      { value: 'token_classification', label: 'Token Classification (NER)' },
      { value: 'text_generation', label: 'Text Generation' },
      { value: 'question_answering', label: 'Question Answering' },
      { value: 'summarization', label: 'Summarization' },
      { value: 'sequence_to_sequence', label: 'Sequence-to-Sequence' },
      { value: 'text_embedding', label: 'Text Embedding' },
    ],
  },
  {
    group: 'Vision',
    items: [
      { value: 'image_classification', label: 'Image Classification' },
      { value: 'object_detection', label: 'Object Detection' },
      { value: 'image_segmentation', label: 'Image Segmentation' },
      { value: 'image_captioning', label: 'Image Captioning' },
    ],
  },
  {
    group: 'Audio',
    items: [
      { value: 'audio_classification', label: 'Audio Classification' },
      { value: 'automatic_speech_recognition', label: 'Speech Recognition (ASR)' },
      { value: 'audio_segmentation', label: 'Audio Segmentation' },
      { value: 'audio_captioning', label: 'Audio Captioning' },
    ],
  },
  {
    group: 'Tabular',
    items: [
      { value: 'tabular_regression', label: 'Tabular Regression' },
      { value: 'tabular_classification', label: 'Tabular Classification' },
      { value: 'tabular_clustering', label: 'Tabular Clustering' },
      { value: 'tabular_anomaly_detection', label: 'Anomaly Detection' },
    ],
  },
]

/** Human-readable task label */
function taskLabel(task: string): string {
  for (const group of TASK_OPTIONS) {
    const item = group.items.find((i) => i.value === task)
    if (item) return item.label
  }
  return task
}

/** Modality color */
function taskColor(task: string): string {
  if (
    task.startsWith('text_') ||
    task === 'question_answering' ||
    task === 'summarization' ||
    task === 'sequence_to_sequence'
  )
    return 'blue'
  if (task.startsWith('image_') || task === 'object_detection') return 'green'
  if (task.startsWith('audio_') || task === 'automatic_speech_recognition') return 'orange'
  if (task.startsWith('tabular_')) return 'grape'
  return 'gray'
}

const CreateProjectModal = () => {
  const form = useForm({
    initialValues: { name: '', description: '', task: '' as string },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
      task: (v) => (v ? null : 'Please select a task'),
    },
  })

  const createProject = useEdenMutation(api.projects.post, [queries.projects.all.queryKey], {
    onSuccess: async ({ project }) => {
      notifications.show({ title: 'Project created', message: `"${project.name}" is ready`, color: 'green' })
      form.reset()
      modals.closeAll()
    },
    onError: (error) => {
      notifications.show({
        title: 'Error',
        message: typeof error.value === 'string' ? error.value : (error.value?.message ?? 'Failed to create project'),
        color: 'red',
      })
    },
  })

  return (
    <form onSubmit={form.onSubmit((values) => createProject.mutate(values as any))}>
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

const UpdateProjectModal = ({
  projectId,
  projectName,
  projectDescription,
}: {
  projectId: string
  projectName: string
  projectDescription: string | null
}) => {
  const form = useForm({
    initialValues: { name: projectName, description: projectDescription },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
    },
  })

  const updateProject = useEdenMutation(api.projects({ projectId }).patch, [queries.projects.all.queryKey], {
    onSuccess: async ({ project }) => {
      notifications.show({ title: 'Project updated', message: `"${project.name}" has been updated`, color: 'green' })
      form.reset()
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
  })

  return (
    <form onSubmit={form.onSubmit((values) => updateProject.mutate(values))}>
      <Stack gap="md">
        <TextInput label="Project name" placeholder="e.g. Traffic Signs" {...form.getInputProps('name')} />
        <Textarea
          label="Description"
          placeholder="What is this project about?"
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

  const { data, isLoading } = useProjects()
  const setActiveProject = useProjectStore((s) => s.setActiveProject)

  const projects = data?.projects ?? []

  useEffect(() => {
    setActiveProject(null)
  }, [setActiveProject])

  const openProject = (projectId: string) => {
    setLocation(`/project/${projectId}`)
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
      <Group justify="space-between" mb="xl">
        <div>
          <Title order={2}>Projects</Title>
          <Text size="sm" c="dimmed" mt={4}>
            Manage your machine learning projects
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
              Create your first project to get started with AI training.
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
              onClick={() => openProject(project.id)}
            >
              <Stack gap="sm">
                <Group justify="space-between">
                  <Title order={5}>{project.name}</Title>
                  <Group gap="xs">
                    <Badge variant="light" color={taskColor(project.task)} size="sm">
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
  )
}

import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Group,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  ArrowRightIcon,
  BrainIcon,
  CrosshairIcon,
  DatabaseIcon,
  PencilSimpleIcon,
  StackIcon,
  TagIcon,
} from '@phosphor-icons/react'
import { api } from '@public/lib/api'
import { assert } from '@public/lib/assert'
import { MODALITY_COLORS } from '@public/lib/constants'
import { useEdenMutation } from '@public/lib/eden-query'
import { queries } from '@public/queries'
import type { ProjectDetail } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { isClassificationTask } from '@server/lib/tasks'
import { useLocation, useParams } from 'wouter'

const BASE_WORKFLOW_STEPS = [
  {
    label: 'Data',
    desc: 'Upload items and build a labeled pool for training',
    icon: DatabaseIcon,
    path: '/data',
    color: 'primary',
  },
  {
    label: 'Dataset',
    desc: 'Assign splits and create version snapshots for training',
    icon: StackIcon,
    path: '/dataset',
    color: 'blue',
  },
  {
    label: 'Training',
    desc: 'Train models on dataset versions and monitor progress',
    icon: BrainIcon,
    path: '/training',
    color: 'teal',
  },
  {
    label: 'Inference',
    desc: 'Test trained models and export weights for deployment',
    icon: CrosshairIcon,
    path: '/inference',
    color: 'orange',
  },
]

const CLASSES_STEP = {
  label: 'Classes',
  desc: 'Define label classes for classification annotations',
  icon: TagIcon,
  path: '/classes',
  color: 'violet',
}

const UpdateProjectModal = ({ project }: { project: ProjectDetail }) => {
  const form = useForm({
    initialValues: { name: project.name, description: project.description },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
    },
  })

  const updateProject = useEdenMutation(
    api.projects({ projectId: project.id }).patch,
    [queries.projects.all.queryKey, queries.projects.detail(project.id).queryKey],
    {
      onSuccess: ({ project }) => {
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
    },
  )

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

export function ProjectPage() {
  const params = useParams<{ id: string }>()
  const [, setLocation] = useLocation()
  const activeProject = useProjectStore((s) => s.activeProject)

  const dataset = activeProject?.dataset
  const versionCount = (dataset?.versions?.length ?? 0) + (dataset?.draft ? 1 : 0)

  const WORKFLOW_STEPS = isClassificationTask(activeProject?.task)
    ? [BASE_WORKFLOW_STEPS[0], CLASSES_STEP, ...BASE_WORKFLOW_STEPS.slice(1)]
    : BASE_WORKFLOW_STEPS

  const handleEditClick = () => {
    assert(activeProject)

    modals.open({
      title: 'Edit project',
      children: <UpdateProjectModal project={activeProject} />,
    })
  }

  return (
    <Box>
      <Stack gap="xl">
        {/* Header */}
        <div>
          <Group gap="xs">
            <Title order={2}>{activeProject?.name ?? 'Project'}</Title>
            {activeProject && (
              <ActionIcon variant="subtle" color="gray" onClick={handleEditClick} size="lg">
                <PencilSimpleIcon size={20} />
              </ActionIcon>
            )}
          </Group>
          <Text size="sm" c="dimmed" mt={4}>
            {activeProject?.description ?? 'Project overview and workflow'}
          </Text>
          <Group gap="xs" mt="xs">
            {activeProject?.task && (
              <Badge variant="light" size="sm">
                {activeProject.task.replace(/_/g, ' ')}
              </Badge>
            )}
            {dataset && (
              <Badge variant="light" color={MODALITY_COLORS[dataset.modality]} size="sm">
                {dataset.modality}
              </Badge>
            )}
          </Group>
        </div>

        {/* Stats */}
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="lg">
          <Card withBorder padding="lg" radius="md">
            <Group>
              <ThemeIcon size="lg" variant="light" color="primary">
                <DatabaseIcon size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  Dataset Versions
                </Text>
                <Text size="xl" fw={700}>
                  {versionCount}
                </Text>
              </div>
            </Group>
          </Card>
          <Card withBorder padding="lg" radius="md">
            <Group>
              <ThemeIcon size="lg" variant="light" color="teal">
                <BrainIcon size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  Training Runs
                </Text>
                <Text size="xl" fw={700}>
                  {activeProject?.runCount ?? 0}
                </Text>
              </div>
            </Group>
          </Card>
          <Card withBorder padding="lg" radius="md">
            <Group>
              <ThemeIcon size="lg" variant="light" color="violet">
                <StackIcon size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  Modality
                </Text>
                <Text size="sm" fw={500} tt="capitalize">
                  {dataset?.modality ?? '—'}
                </Text>
              </div>
            </Group>
          </Card>
        </SimpleGrid>

        {/* Workflow */}
        <div>
          <Title order={4} mb="md">
            Workflow
          </Title>
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
            {WORKFLOW_STEPS.map((step, i) => (
              <Card
                key={step.label}
                withBorder
                padding="lg"
                radius="md"
                className="card-elevated"
                style={{ cursor: 'pointer' }}
                onClick={() => setLocation(`/project/${params.id}${step.path}`)}
              >
                <Group justify="space-between" mb="sm">
                  <Group gap="sm">
                    <ThemeIcon size="md" variant="light" color={step.color}>
                      <step.icon size={18} />
                    </ThemeIcon>
                    <Text fw={600}>{step.label}</Text>
                  </Group>
                  <Badge variant="dot" color={step.color} size="sm">
                    Step {i + 1}
                  </Badge>
                </Group>
                <Text size="sm" c="dimmed">
                  {step.desc}
                </Text>
                <Group justify="flex-end" mt="sm">
                  <ArrowRightIcon size={16} />
                </Group>
              </Card>
            ))}
          </SimpleGrid>
        </div>
      </Stack>
    </Box>
  )
}

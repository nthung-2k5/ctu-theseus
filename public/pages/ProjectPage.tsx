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
  MagicWandIcon,
  PencilSimpleIcon,
  TabsIcon,
  TagChevronIcon,
  TagIcon,
} from '@phosphor-icons/react'
import { api } from '@public/lib/api'
import { assert } from '@public/lib/assert'
import { useEdenMutation } from '@public/lib/eden-query'
import { queries } from '@public/queries'
import type { Project } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { useLocation, useParams } from 'wouter'

const WORKFLOW_STEPS = [
  {
    label: 'Dataset',
    desc: 'Upload images and define classes',
    icon: DatabaseIcon,
    path: '/dataset',
    color: 'primary',
  },
  {
    label: 'Labeling',
    desc: 'Label images manually or auto-detect',
    icon: TagIcon,
    path: '/labeling',
    color: 'secondary',
  },
  {
    label: 'Augmentation',
    desc: 'Configure data augmentation',
    icon: MagicWandIcon,
    path: '/augmentation',
    color: 'violet',
  },
  { label: 'Training', desc: 'Select model and train', icon: BrainIcon, path: '/training', color: 'teal' },
  {
    label: 'Inference',
    desc: 'Test model and export weights',
    icon: CrosshairIcon,
    path: '/inference',
    color: 'orange',
  },
]

const UpdateProjectModal = ({ project }: { project: Project }) => {
  const form = useForm({
    initialValues: { name: project.name, description: project.description },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
    },
  })

  const updateProject = useEdenMutation(api.projects({ projectId: project.id }).patch, [queries.projects.all.queryKey, queries.projects.detail(project.id).queryKey], {
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

export function ProjectPage() {
  const params = useParams<{ id: string }>()
  const [, setLocation] = useLocation()
  const activeProject = useProjectStore((s) => s.activeProject)

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
                  Dataset Images
                </Text>
                <Text size="xl" fw={700}>
                  {activeProject?.imageCount}
                </Text>
              </div>
            </Group>
          </Card>
          <Card withBorder padding="lg" radius="md">
            <Group>
              <ThemeIcon size="lg" variant="light" color="secondary">
                <TagChevronIcon size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  Classes
                </Text>
                <Text size="xl" fw={700}>
                  {activeProject?.classCount}
                </Text>
              </div>
            </Group>
          </Card>
          <Card withBorder padding="lg" radius="md">
            <Group>
              <ThemeIcon size="lg" variant="light" color="violet">
                <TabsIcon size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  Versions
                </Text>
                <Text size="xl" fw={700}>
                  {activeProject?.versionCount}
                </Text>
              </div>
            </Group>
          </Card>
          {/* <Card withBorder padding="lg" radius="md">
            <Group>
              <ThemeIcon size="lg" variant="light" color="violet">
                <ChartBarIcon size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  Dataset Split
                </Text>
                <Text size="sm" fw={500}>
                  {splitConfig.train}/{splitConfig.validate}/{splitConfig.test}
                </Text>
              </div>
            </Group>
            <Progress.Root size="sm" mt="sm">
              <Progress.Section value={splitConfig.train} color="green" />
              <Progress.Section value={splitConfig.validate} color="yellow" />
              <Progress.Section value={splitConfig.test} color="red" />
            </Progress.Root>
          </Card> */}
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

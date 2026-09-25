import {
  Badge,
  Button,
  Divider,
  Group,
  Paper,
  Select,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  Textarea,
  TextInput,
  ThemeIcon,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { PlusIcon } from '@phosphor-icons/react'
import { LinkCard, PageHeader, QueryBoundary } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import type { DatasetModality, ProjectTask } from '@public/lib/api/enums'
import { getCreateProjectMutationOptions } from '@public/lib/api/generated/projects/projects'
import { useListTrainingBackends } from '@public/lib/api/generated/training/training'
import { formatDate } from '@public/lib/format'
import { MODALITY_META } from '@public/lib/modality'
import { invalidateProjectList, useProjects } from '@public/lib/queries'
import { taskRegistry } from '@public/lib/tasks'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'

/**
 * Tasks grouped by modality for selection, enabled iff some installed and available trainer
 * backend actually supports them right now (GET /training-backends) — see `useTaskOptions`.
 */
function useTaskOptions() {
  const { data } = useListTrainingBackends()
  const trainableTaskIds = useMemo(() => {
    const ids = new Set<string>()
    for (const backend of data?.backends ?? []) {
      if (backend.available) for (const taskId of backend.supportedTasks) ids.add(taskId)
    }
    return ids
  }, [data])

  const options = useMemo(
    () =>
      (Object.keys(MODALITY_META) as DatasetModality[]).map((modality) => ({
        group: MODALITY_META[modality].label,
        items: Object.values(taskRegistry)
          .filter((d) => d.modality === modality)
          .map((d) => ({
            value: d.id,
            label: trainableTaskIds.has(d.id) ? d.label : `${d.label} (coming soon)`,
            disabled: !trainableTaskIds.has(d.id),
          })),
      })),
    [trainableTaskIds],
  )

  return { options, trainableTaskIds }
}

const QUICK_STARTERS: {
  title: string
  task: ProjectTask
  defaultName: string
  defaultDesc: string
  modality: DatasetModality
}[] = [
  {
    title: 'Image Classifier',
    task: 'image_classification',
    defaultName: 'Image Classification Model',
    defaultDesc: 'Train a vision backbone on image categories.',
    modality: 'vision',
  },
  {
    title: 'Text Classifier',
    task: 'text_classification',
    defaultName: 'Text Sentiment & Topic Model',
    defaultDesc: 'Categorize text data using modern transformer encoders.',
    modality: 'text',
  },
  {
    title: 'Tabular Classifier',
    task: 'tabular_classification',
    defaultName: 'Tabular Predictor',
    defaultDesc: 'Classify structured tabular records and multi-feature data.',
    modality: 'tabular',
  },
  {
    title: 'Audio Classifier',
    task: 'audio_classification',
    defaultName: 'Audio Classifier',
    defaultDesc: 'Classify audio waveforms and acoustic event recordings.',
    modality: 'audio',
  },
]

function taskLabel(task: string): string {
  return taskRegistry[task as ProjectTask]?.label ?? task
}

const CreateProjectModal = () => {
  const form = useForm({
    initialValues: { name: '', description: '', task: 'image_classification' as string },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
      task: (v) => (v ? null : 'Please select a task'),
    },
  })

  const queryClient = useQueryClient()

  const selectedTaskDescriptor = form.values.task ? taskRegistry[form.values.task as ProjectTask] : undefined
  const selectedModality = selectedTaskDescriptor?.modality
  const { options: taskOptions, trainableTaskIds } = useTaskOptions()
  const isTrainable = !!selectedTaskDescriptor && trainableTaskIds.has(selectedTaskDescriptor.id)

  const createProject = useMutation({
    ...getCreateProjectMutationOptions(),
    onSuccess: ({ project }) => {
      invalidateProjectList(queryClient)
      notifications.show({ title: 'Project created', message: `"${project.name}" is ready`, color: 'green' })
      form.reset()
      modals.closeAll()
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to create project'), color: 'red' })
    },
  })

  const applyStarter = (starter: (typeof QUICK_STARTERS)[number]) => {
    form.setValues({
      name: starter.defaultName,
      description: starter.defaultDesc,
      task: starter.task,
    })
  }

  return (
    <form
      onSubmit={form.onSubmit((values) =>
        createProject.mutate({
          data: { name: values.name, description: values.description, task: values.task as ProjectTask },
        }),
      )}
    >
      <Stack gap="md">
        {/* Quick starter presets */}
        <div>
          <Text size="xs" fw={600} c="dimmed" tt="uppercase" mb={6}>
            Quick Starters
          </Text>
          <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="xs">
            {QUICK_STARTERS.map((s) => {
              const meta = MODALITY_META[s.modality]
              const Icon = meta.icon
              const isSelected = form.values.task === s.task
              return (
                <Paper
                  key={s.task}
                  withBorder
                  p="xs"
                  radius="md"
                  onClick={() => applyStarter(s)}
                  style={{
                    cursor: 'pointer',
                    borderColor: isSelected ? `var(--mantine-color-${meta.color}-filled)` : undefined,
                    backgroundColor: isSelected ? `var(--mantine-color-${meta.color}-light)` : undefined,
                    transition: 'all 150ms ease',
                  }}
                >
                  <Group gap={6} wrap="nowrap">
                    <ThemeIcon size="sm" variant="light" color={meta.color}>
                      <Icon size={14} />
                    </ThemeIcon>
                    <Text size="xs" fw={600} truncate>
                      {s.title}
                    </Text>
                  </Group>
                </Paper>
              )
            })}
          </SimpleGrid>
        </div>

        <Divider />

        <TextInput
          label="Project name"
          placeholder="e.g. Defect Detection Model"
          autoFocus
          required
          {...form.getInputProps('name')}
        />

        <Textarea
          label="Description"
          placeholder="Describe your dataset, training objective, or model targets (optional)..."
          autosize
          minRows={3}
          maxRows={5}
          {...form.getInputProps('description')}
        />

        <Select
          label="ML Task"
          placeholder="Select ML task"
          data={taskOptions}
          searchable
          required
          {...form.getInputProps('task')}
        />

        {selectedTaskDescriptor && selectedModality && (
          <Paper withBorder p="sm" radius="md" bg="var(--mantine-color-default-hover)">
            <Group justify="space-between" align="center" mb={4}>
              <Group gap="xs">
                <Badge variant="light" color={MODALITY_META[selectedModality].color} size="sm">
                  {MODALITY_META[selectedModality].label}
                </Badge>
                <Text size="xs" fw={600}>
                  {selectedTaskDescriptor.label}
                </Text>
              </Group>
              <Badge variant="dot" color={isTrainable ? 'green' : 'gray'} size="xs">
                {isTrainable ? 'Trainer Ready' : 'Planned'}
              </Badge>
            </Group>
            <Text size="xs" c="dimmed">
              Payload: <strong>{selectedTaskDescriptor.itemSpec.payload}</strong> &bull; Annotation:{' '}
              <strong>{selectedTaskDescriptor.annotation.type}</strong>
            </Text>
          </Paper>
        )}

        <Group justify="flex-end" mt="xs">
          <Button variant="subtle" onClick={modals.closeAll}>
            Cancel
          </Button>
          <Button type="submit" loading={createProject.isPending} leftSection={<PlusIcon size={16} />}>
            Create Project
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

export function DashboardPage() {
  const { data, isLoading, isError, refetch } = useProjects()
  const projects = [...(data?.projects ?? [])].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  )

  const openCreateProjectModal = () => {
    modals.open({
      title: 'Create new project',
      centered: true,
      size: 'lg',
      children: <CreateProjectModal />,
    })
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Projects"
        description="Each project is one task on one modality."
        actions={
          <Button leftSection={<PlusIcon size={15} />} onClick={openCreateProjectModal}>
            New project
          </Button>
        }
      />

      <QueryBoundary
        isLoading={isLoading}
        isError={isError}
        onRetry={() => refetch()}
        loadingFallback={
          <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }} spacing="sm">
            {Array.from({ length: 6 }, (_, i) => `skeleton-${i}`).map((key) => (
              <Paper key={key} p="md">
                <Stack gap="xs">
                  <Skeleton height={16} width="50%" radius="sm" />
                  <Skeleton height={12} width="90%" radius="sm" />
                  <Skeleton height={12} width="60%" radius="sm" />
                </Stack>
              </Paper>
            ))}
          </SimpleGrid>
        }
      >
        {projects.length === 0 && (
          <Text c="dimmed" ta="center" py="xl">
            No projects yet. Create one to get started.
          </Text>
        )}
        <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }} spacing="sm">
          {projects.map((project) => {
            const modality = taskRegistry[project.task]?.modality ?? 'vision'
            const meta = MODALITY_META[modality] ?? MODALITY_META.vision
            return (
              <LinkCard
                key={project.id}
                to="/project/$projectId"
                params={{ projectId: project.id }}
                withBorder
                padding="md"
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <Group justify="space-between" mb={4} wrap="nowrap">
                  <Text fw={600} truncate>
                    {project.name}
                  </Text>
                  <Badge color={meta.color}>{meta.label}</Badge>
                </Group>
                <Badge variant="outline" color="gray" mb="xs">
                  {taskLabel(project.task)}
                </Badge>
                <Text size="xs" c="dimmed" lineClamp={2} mih={32}>
                  {project.description || 'No description'}
                </Text>
                <Text size="xs" c="dimmed" className="tnum" mt="sm">
                  Created {formatDate(project.createdAt)}
                </Text>
              </LinkCard>
            )
          })}
        </SimpleGrid>
      </QueryBoundary>
    </div>
  )
}

import {
  Badge,
  Button,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
  ThemeIcon,
  UnstyledButton,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { PlusIcon } from '@phosphor-icons/react'
import { apiErrorMessage } from '@public/lib/api/client'
import type { DatasetModality, ProjectTask } from '@public/lib/api/enums'
import { getCreateProjectMutationOptions } from '@public/lib/api/generated/projects/projects'
import { useListTrainingBackends } from '@public/lib/api/generated/training/training'
import { MODALITY_META } from '@public/lib/modality'
import { invalidateProjectList } from '@public/lib/queries'
import { getTaskMeta } from '@public/lib/task-meta'
import { taskRegistry } from '@public/lib/tasks'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'

const MODALITIES = Object.keys(MODALITY_META) as DatasetModality[]

/** Ids of the tasks some installed, available trainer backend can train right now (GET /training-backends). */
function useTrainableTaskIds(): Set<string> {
  const { data } = useListTrainingBackends()
  return useMemo(() => {
    const ids = new Set<string>()
    for (const backend of data?.backends ?? []) {
      if (backend.available) for (const taskId of backend.supportedTasks) ids.add(taskId)
    }
    return ids
  }, [data])
}

function selectedStyle(color: string, selected: boolean) {
  return {
    borderColor: selected ? `var(--mantine-color-${color}-filled)` : undefined,
    backgroundColor: selected ? `var(--mantine-color-${color}-light)` : undefined,
    transition: 'all 150ms ease',
  }
}

export function openCreateProjectModal() {
  modals.open({
    title: 'Create new project',
    fullScreen: true,
    children: <CreateProjectModal />,
    styles: {
      title: { fontSize: 'var(--mantine-font-size-xl)', fontWeight: 700 },
      header: { minHeight: 72 },
      body: { height: 'calc(100% - 72px)' },
    },
  })
}

function CreateProjectModal() {
  const form = useForm({
    initialValues: { name: '', description: '', modality: 'vision' as DatasetModality, task: 'image_classification' },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
      task: (v) => (v ? null : 'Pick a task for this project'),
    },
  })

  const queryClient = useQueryClient()
  const trainableTaskIds = useTrainableTaskIds()

  const tasks = useMemo(
    () => Object.values(taskRegistry).filter((d) => d.modality === form.values.modality),
    [form.values.modality],
  )

  const createProject = useMutation({
    ...getCreateProjectMutationOptions(),
    onSuccess: ({ project }) => {
      invalidateProjectList(queryClient)
      notifications.show({ title: 'Project created', message: `"${project.name}" is ready`, color: 'green' })
      modals.closeAll()
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to create project'), color: 'red' })
    },
  })

  const pickModality = (modality: DatasetModality) => {
    if (modality === form.values.modality) return
    const first = Object.values(taskRegistry).find((d) => d.modality === modality && trainableTaskIds.has(d.id))
    form.setValues({ modality, task: first?.id ?? '' })
  }

  return (
    <form
      className="h-full"
      onSubmit={form.onSubmit((values) =>
        createProject.mutate({
          data: { name: values.name, description: values.description, task: values.task as ProjectTask },
        }),
      )}
    >
      <div className="grid h-full grid-cols-1 gap-8 lg:grid-cols-[minmax(320px,420px)_1fr]">
        <Paper withBorder p="lg" radius="md" className="min-h-0 overflow-y-auto">
          <Stack gap="md">
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
              minRows={4}
              maxRows={8}
              {...form.getInputProps('description')}
            />

            <div>
              <Text size="sm" fw={500} mb={4}>
                Modality
              </Text>
              <SimpleGrid cols={2} spacing="xs" role="radiogroup" aria-label="Modality">
                {MODALITIES.map((modality) => {
                  const meta = MODALITY_META[modality]
                  const Icon = meta.icon
                  const selected = form.values.modality === modality
                  return (
                    <UnstyledButton
                      key={modality}
                      role="radio"
                      aria-checked={selected}
                      onClick={() => pickModality(modality)}
                    >
                      <Paper withBorder p="sm" radius="md" style={selectedStyle(meta.color, selected)}>
                        <Group gap="xs" wrap="nowrap">
                          <ThemeIcon variant={selected ? 'filled' : 'light'} color={meta.color}>
                            <Icon size={18} />
                          </ThemeIcon>
                          <Text size="sm" fw={600}>
                            {meta.label}
                          </Text>
                        </Group>
                      </Paper>
                    </UnstyledButton>
                  )
                })}
              </SimpleGrid>
              <Text size="xs" c="dimmed" mt={6}>
                {MODALITY_META[form.values.modality].description}
              </Text>
            </div>
          </Stack>
        </Paper>

        <Paper withBorder radius="md" className="flex min-h-0 flex-col">
          <Text size="sm" fw={500} px="lg" pt="lg" pb="xs">
            Task
          </Text>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-3">
            <Stack gap="xs" role="radiogroup" aria-label="Task">
              {tasks.map((task) => {
                const { icon: Icon, description } = getTaskMeta(task.id)
                const color = MODALITY_META[task.modality].color
                const selected = form.values.task === task.id
                const trainable = trainableTaskIds.has(task.id)
                return (
                  <UnstyledButton
                    key={task.id}
                    role="radio"
                    aria-checked={selected}
                    disabled={!trainable}
                    onClick={() => form.setFieldValue('task', task.id)}
                    style={{ opacity: trainable ? 1 : 0.55, cursor: trainable ? 'pointer' : 'not-allowed' }}
                  >
                    <Paper withBorder p="md" radius="md" style={selectedStyle(color, selected)}>
                      <Group gap="md" wrap="nowrap" align="flex-start">
                        <ThemeIcon size={40} radius="md" variant={selected ? 'filled' : 'light'} color={color}>
                          <Icon size={22} />
                        </ThemeIcon>
                        <div className="min-w-0 flex-1">
                          <Group gap="xs" justify="space-between" wrap="nowrap">
                            <Text fw={700}>{task.label}</Text>
                            {!trainable ? (
                              <Badge variant="light" color="gray" size="xs">
                                Coming soon
                              </Badge>
                            ) : task.status === 'experimental' ? (
                              <Badge variant="light" color="yellow" size="xs">
                                Experimental
                              </Badge>
                            ) : null}
                          </Group>
                          {description && (
                            <Text size="sm" c="dimmed">
                              {description}
                            </Text>
                          )}
                        </div>
                      </Group>
                    </Paper>
                  </UnstyledButton>
                )
              })}
            </Stack>
          </div>
          <Group justify="space-between" p="md" className="border-t border-(--mantine-color-default-border)">
            <Text size="xs" c="red">
              {form.errors.task}
            </Text>
            <Group>
              <Button variant="subtle" onClick={modals.closeAll}>
                Cancel
              </Button>
              <Button
                type="submit"
                loading={createProject.isPending}
                disabled={form.values.name.trim().length === 0}
                leftSection={<PlusIcon size={16} />}
              >
                Create Project
              </Button>
            </Group>
          </Group>
        </Paper>
      </div>
    </form>
  )
}

import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Group,
  Paper,
  SegmentedControl,
  Select,
  SimpleGrid,
  Skeleton,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  ThemeIcon,
  Title,
  Tooltip,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  ArrowRightIcon,
  ArrowsDownUpIcon,
  CalendarBlankIcon,
  ChatTextIcon,
  FolderSimpleIcon,
  ImageSquareIcon,
  ListBulletsIcon,
  MagnifyingGlassIcon,
  PencilSimpleIcon,
  PlusIcon,
  SquaresFourIcon,
  TableIcon,
  TrashIcon,
  WaveformIcon,
  XIcon,
} from '@phosphor-icons/react'
import { UpdateProjectModal } from '@public/components/UpdateProjectModal'
import { confirmDelete, EmptyState, PageHeader, QueryBoundary } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import type { DatasetModality, ProjectTask } from '@public/lib/api/enums'
import {
  deleteProject as deleteProjectRequest,
  getCreateProjectMutationOptions,
} from '@public/lib/api/generated/projects/projects'
import { useListTrainingBackends } from '@public/lib/api/generated/training/training'
import { formatDate } from '@public/lib/format'
import { invalidateProjectList, useProjects } from '@public/lib/queries'
import { taskRegistry } from '@public/lib/tasks'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useMemo, useState } from 'react'

interface ModalityMeta {
  label: string
  color: string
  icon: typeof ImageSquareIcon
  description: string
}

const MODALITY_META: Record<DatasetModality, ModalityMeta> = {
  vision: {
    label: 'Vision',
    color: 'green',
    icon: ImageSquareIcon,
    description: 'Image classification, detection & visual understanding',
  },
  text: {
    label: 'Text',
    color: 'blue',
    icon: ChatTextIcon,
    description: 'NLP, sequence tagging, classification & generation',
  },
  audio: {
    label: 'Audio',
    color: 'orange',
    icon: WaveformIcon,
    description: 'Sound classification & acoustic event labeling',
  },
  tabular: {
    label: 'Tabular',
    color: 'grape',
    icon: TableIcon,
    description: 'Structured records, numerical & category prediction',
  },
}

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
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data, isLoading, isError, refetch } = useProjects()
  const projects = data?.projects ?? []

  // Interactive filtering, search, and view states
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedModality, setSelectedModality] = useState<string>('all')
  const [sortBy, setSortBy] = useState<string>('newest')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  const openProject = (projectId: string) => {
    navigate({ to: '/project/$projectId', params: { projectId } })
  }

  const openCreateProjectModal = () => {
    modals.open({
      title: 'Create new project',
      centered: true,
      size: 'lg',
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

  const deleteProject = useMutation({
    mutationFn: async (projectId: string) => {
      await deleteProjectRequest(projectId)
    },
    onSuccess: () => {
      invalidateProjectList(queryClient)
      notifications.show({ title: 'Project deleted', message: 'The project has been removed', color: 'green' })
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to delete project', color: 'red' })
    },
  })

  const handleDeleteClick = (
    e: React.MouseEvent<HTMLButtonElement, MouseEvent>,
    projectId: string,
    projectName: string,
  ) => {
    e.stopPropagation()
    confirmDelete({
      title: 'Delete project',
      message: (
        <>
          Are you sure you want to delete <strong>{projectName}</strong>? This permanently removes its dataset,
          versions, items, and training runs. This cannot be undone.
        </>
      ),
      onConfirm: () => deleteProject.mutate(projectId),
    })
  }

  // Modality statistics breakdown
  const modalityCounts = useMemo(() => {
    const counts: Record<string, number> = {
      all: projects.length,
      vision: 0,
      text: 0,
      audio: 0,
      tabular: 0,
    }
    for (const p of projects) {
      const modality = taskRegistry[p.task]?.modality
      if (modality && counts[modality] !== undefined) {
        counts[modality]++
      }
    }
    return counts
  }, [projects])

  // Filtered & sorted projects list
  const filteredProjects = useMemo(() => {
    let result = [...projects]

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim()
      result = result.filter((p) => {
        const name = p.name.toLowerCase()
        const desc = (p.description ?? '').toLowerCase()
        const task = p.task.toLowerCase()
        const taskObj = taskRegistry[p.task]
        const taskLbl = (taskObj?.label ?? '').toLowerCase()
        const modality = (taskObj?.modality ?? '').toLowerCase()
        return name.includes(q) || desc.includes(q) || task.includes(q) || taskLbl.includes(q) || modality.includes(q)
      })
    }

    if (selectedModality !== 'all') {
      result = result.filter((p) => {
        const taskObj = taskRegistry[p.task]
        return taskObj?.modality === selectedModality
      })
    }

    result.sort((a, b) => {
      if (sortBy === 'newest') {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      }
      if (sortBy === 'oldest') {
        return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
      }
      if (sortBy === 'name-asc') {
        return a.name.localeCompare(b.name)
      }
      if (sortBy === 'name-desc') {
        return b.name.localeCompare(a.name)
      }
      return 0
    })

    return result
  }, [projects, searchQuery, selectedModality, sortBy])

  return (
    <Box>
      {/* ─── Page Header ─── */}
      <PageHeader
        title="AI Projects"
        description="Create, annotate, train, and manage machine learning pipelines across modalities"
        actions={
          <Button leftSection={<PlusIcon size={18} />} onClick={openCreateProjectModal}>
            New project
          </Button>
        }
      />

      {/* ─── Search, Filter, Sort & View Control Bar ─── */}
      {!isLoading && projects.length > 0 && (
        <Paper withBorder p="md" radius="md" mt="xl">
          <Stack gap="sm">
            <Group justify="space-between" wrap="wrap">
              {/* Search bar */}
              <TextInput
                placeholder="Search projects by name, description, or task..."
                leftSection={<MagnifyingGlassIcon size={16} />}
                rightSection={
                  searchQuery ? (
                    <ActionIcon size="xs" variant="subtle" color="gray" onClick={() => setSearchQuery('')}>
                      <XIcon size={14} />
                    </ActionIcon>
                  ) : null
                }
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.currentTarget.value)}
                style={{ flexGrow: 1, minWidth: 260 }}
              />

              <Group gap="sm">
                {/* Sort dropdown */}
                <Select
                  leftSection={<ArrowsDownUpIcon size={16} />}
                  value={sortBy}
                  onChange={(v) => setSortBy(v ?? 'newest')}
                  data={[
                    { value: 'newest', label: 'Newest first' },
                    { value: 'oldest', label: 'Oldest first' },
                    { value: 'name-asc', label: 'Name (A to Z)' },
                    { value: 'name-desc', label: 'Name (Z to A)' },
                  ]}
                  w={180}
                  allowDeselect={false}
                />

                {/* View switcher */}
                <SegmentedControl
                  value={viewMode}
                  onChange={(v) => setViewMode(v as 'grid' | 'list')}
                  data={[
                    {
                      value: 'grid',
                      label: (
                        <Group gap={4} justify="center" wrap="nowrap">
                          <SquaresFourIcon size={16} />
                          <Text size="sm" visibleFrom="xs">
                            Grid
                          </Text>
                        </Group>
                      ),
                    },
                    {
                      value: 'list',
                      label: (
                        <Group gap={4} justify="center" wrap="nowrap">
                          <ListBulletsIcon size={16} />
                          <Text size="sm" visibleFrom="xs">
                            List
                          </Text>
                        </Group>
                      ),
                    },
                  ]}
                />
              </Group>
            </Group>

            {/* Modality filter chips */}
            <Group gap="xs" wrap="wrap">
              <Text size="xs" fw={600} c="dimmed" mr="xs">
                Modality:
              </Text>
              <Button
                size="xs"
                variant={selectedModality === 'all' ? 'filled' : 'light'}
                color="gray"
                radius="xl"
                onClick={() => setSelectedModality('all')}
              >
                All ({modalityCounts.all})
              </Button>
              {(Object.keys(MODALITY_META) as DatasetModality[]).map((modality) => {
                const meta = MODALITY_META[modality]
                const Icon = meta.icon
                const count = modalityCounts[modality] ?? 0
                const isSelected = selectedModality === modality
                return (
                  <Button
                    key={modality}
                    size="xs"
                    variant={isSelected ? 'filled' : 'light'}
                    color={meta.color}
                    radius="xl"
                    leftSection={<Icon size={14} />}
                    onClick={() => setSelectedModality(modality)}
                    disabled={count === 0}
                  >
                    {meta.label} ({count})
                  </Button>
                )
              })}
            </Group>
          </Stack>
        </Paper>
      )}

      {/* ─── Projects List / Grid Body ─── */}
      <Box mt="xl">
        <QueryBoundary
          isLoading={isLoading}
          isError={isError}
          onRetry={() => refetch()}
          loadingFallback={
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg">
              {Array.from({ length: 6 }, (_, i) => `skeleton-${i}`).map((key) => (
                <Card key={key} withBorder padding="lg" radius="md">
                  <Stack gap="md">
                    <Group justify="space-between">
                      <Skeleton height={24} width="50%" radius="sm" />
                      <Skeleton height={20} width={70} radius="xl" />
                    </Group>
                    <Skeleton height={14} width="90%" radius="sm" />
                    <Skeleton height={14} width="60%" radius="sm" />
                    <Divider />
                    <Group justify="space-between">
                      <Skeleton height={16} width={80} radius="sm" />
                      <Skeleton height={28} width={90} radius="md" />
                    </Group>
                  </Stack>
                </Card>
              ))}
            </SimpleGrid>
          }
        >
          {projects.length === 0 ? (
            <Paper withBorder p="xl" radius="md" style={{ textAlign: 'center' }}>
              <Stack align="center" gap="md" py="xl">
                <ThemeIcon size={64} radius="xl" variant="light" color="primary">
                  <FolderSimpleIcon size={36} />
                </ThemeIcon>
                <div>
                  <Title order={3}>Welcome to CTU Theseus</Title>
                  <Text size="sm" c="dimmed" mt={4} maw={500} mx="auto">
                    Build and train machine learning models end-to-end. Create your first project to start uploading
                    data, defining labels, and training neural networks.
                  </Text>
                </div>
                <Button leftSection={<PlusIcon size={18} />} onClick={openCreateProjectModal} mt="sm">
                  Create your first project
                </Button>
              </Stack>
            </Paper>
          ) : filteredProjects.length === 0 ? (
            <Paper withBorder p="xl" radius="md">
              <EmptyState
                icon={MagnifyingGlassIcon}
                title="No projects match your filter"
                description={`No projects found matching "${searchQuery}" in ${selectedModality === 'all' ? 'all modalities' : selectedModality}.`}
                action={
                  <Button
                    variant="light"
                    onClick={() => {
                      setSearchQuery('')
                      setSelectedModality('all')
                    }}
                  >
                    Clear search & filters
                  </Button>
                }
              />
            </Paper>
          ) : viewMode === 'grid' ? (
            /* ─── Grid Cards View ─── */
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg">
              {filteredProjects.map((project) => {
                const descriptor = taskRegistry[project.task]
                const modality = descriptor?.modality ?? 'vision'
                const meta = MODALITY_META[modality] ?? MODALITY_META.vision
                const ModalityIcon = meta.icon

                return (
                  <Card
                    key={project.id}
                    withBorder
                    padding="lg"
                    radius="md"
                    className="card-elevated"
                    style={{
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                    }}
                    onClick={() => openProject(project.id)}
                  >
                    <Stack gap="sm">
                      {/* Top row: Modality Icon, Task Badge, and Actions */}
                      <Group justify="space-between" align="flex-start" wrap="nowrap">
                        <Group gap="xs" wrap="nowrap">
                          <ThemeIcon size="md" variant="light" color={meta.color} radius="md">
                            <ModalityIcon size={18} />
                          </ThemeIcon>
                          <div>
                            <Badge variant="light" color={meta.color} size="sm">
                              {taskLabel(project.task)}
                            </Badge>
                          </div>
                        </Group>

                        {/* Card Action Menu */}
                        <Group gap={4} wrap="nowrap">
                          <Tooltip label="Edit project">
                            <ActionIcon
                              variant="subtle"
                              color="gray"
                              size="sm"
                              onClick={(e) => openUpdateProjectModal(e, project.id, project.name, project.description)}
                            >
                              <PencilSimpleIcon size={16} />
                            </ActionIcon>
                          </Tooltip>
                          <Tooltip label="Delete project">
                            <ActionIcon
                              variant="subtle"
                              color="red"
                              size="sm"
                              loading={deleteProject.isPending && deleteProject.variables === project.id}
                              onClick={(e) => handleDeleteClick(e, project.id, project.name)}
                            >
                              <TrashIcon size={16} />
                            </ActionIcon>
                          </Tooltip>
                        </Group>
                      </Group>

                      {/* Title & Description */}
                      <div>
                        <Title order={4} lineClamp={1}>
                          {project.name}
                        </Title>
                        <Text size="sm" c="dimmed" lineClamp={2} mt={4} style={{ minHeight: '2.6em' }}>
                          {project.description || (
                            <Text component="span" fs="italic" c="dimmed">
                              No description provided
                            </Text>
                          )}
                        </Text>
                      </div>
                    </Stack>

                    {/* Card Footer: Date & Open Button */}
                    <Box mt="md">
                      <Divider mb="sm" />
                      <Group justify="space-between" align="center">
                        <Group gap={6}>
                          <CalendarBlankIcon size={14} color="var(--mantine-color-dimmed)" />
                          <Text size="xs" c="dimmed">
                            {formatDate(project.createdAt)}
                          </Text>
                        </Group>
                        <Group gap={4} c="primary">
                          <Text size="xs" fw={600}>
                            Open
                          </Text>
                          <ArrowRightIcon size={14} />
                        </Group>
                      </Group>
                    </Box>
                  </Card>
                )
              })}
            </SimpleGrid>
          ) : (
            /* ─── List / Table View ─── */
            <Paper withBorder radius="md">
              <Table highlightOnHover verticalSpacing="sm" horizontalSpacing="md">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Project</Table.Th>
                    <Table.Th>Modality & Task</Table.Th>
                    <Table.Th>Created</Table.Th>
                    <Table.Th style={{ textAlign: 'right' }}>Actions</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {filteredProjects.map((project) => {
                    const descriptor = taskRegistry[project.task]
                    const modality = descriptor?.modality ?? 'vision'
                    const meta = MODALITY_META[modality] ?? MODALITY_META.vision
                    const ModalityIcon = meta.icon

                    return (
                      <Table.Tr key={project.id} style={{ cursor: 'pointer' }} onClick={() => openProject(project.id)}>
                        <Table.Td style={{ minWidth: 200 }}>
                          <Group gap="sm" wrap="nowrap">
                            <ThemeIcon size="md" variant="light" color={meta.color} radius="md">
                              <ModalityIcon size={18} />
                            </ThemeIcon>
                            <div>
                              <Text size="sm" fw={600}>
                                {project.name}
                              </Text>
                              <Text size="xs" c="dimmed" lineClamp={1}>
                                {project.description || 'No description provided'}
                              </Text>
                            </div>
                          </Group>
                        </Table.Td>
                        <Table.Td>
                          <Badge variant="light" color={meta.color} size="sm">
                            {taskLabel(project.task)}
                          </Badge>
                        </Table.Td>
                        <Table.Td>
                          <Text size="xs" c="dimmed">
                            {formatDate(project.createdAt)}
                          </Text>
                        </Table.Td>
                        <Table.Td style={{ textAlign: 'right' }}>
                          <Group gap={4} justify="flex-end" wrap="nowrap">
                            <Tooltip label="Edit project">
                              <ActionIcon
                                variant="subtle"
                                color="gray"
                                onClick={(e) =>
                                  openUpdateProjectModal(e, project.id, project.name, project.description)
                                }
                              >
                                <PencilSimpleIcon size={16} />
                              </ActionIcon>
                            </Tooltip>
                            <Tooltip label="Delete project">
                              <ActionIcon
                                variant="subtle"
                                color="red"
                                loading={deleteProject.isPending && deleteProject.variables === project.id}
                                onClick={(e) => handleDeleteClick(e, project.id, project.name)}
                              >
                                <TrashIcon size={16} />
                              </ActionIcon>
                            </Tooltip>
                            <Button
                              size="xs"
                              variant="light"
                              rightSection={<ArrowRightIcon size={14} />}
                              onClick={(e) => {
                                e.stopPropagation()
                                openProject(project.id)
                              }}
                            >
                              Open
                            </Button>
                          </Group>
                        </Table.Td>
                      </Table.Tr>
                    )
                  })}
                </Table.Tbody>
              </Table>
            </Paper>
          )}
        </QueryBoundary>
      </Box>
    </Box>
  )
}

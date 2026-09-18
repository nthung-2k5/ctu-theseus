import {
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Grid,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Table,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  ArrowRightIcon,
  BrainIcon,
  ChatTextIcon,
  CheckCircleIcon,
  CpuIcon,
  DatabaseIcon,
  ImageSquareIcon,
  ListChecksIcon,
  PencilSimpleIcon,
  PlusIcon,
  StackIcon,
  TableIcon,
  TrashIcon,
  WaveformIcon,
} from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { UpdateProjectModal } from '@public/components/UpdateProjectModal'
import { confirmDelete, StatCard } from '@public/components/ui'
import { rest, useEden } from '@public/lib/api'
import { formatDate } from '@public/lib/format'
import { projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import type { DatasetModality } from '@server/lib/enums'
import { getTaskDescriptor, isClassificationTask } from '@server/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, useNavigate } from '@tanstack/react-router'
import { useMemo } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/')

const MODALITY_CONFIG: Record<
  DatasetModality,
  {
    label: string
    color: string
    icon: typeof ImageSquareIcon
  }
> = {
  vision: { label: 'Vision', color: 'green', icon: ImageSquareIcon },
  text: { label: 'Text', color: 'blue', icon: ChatTextIcon },
  audio: { label: 'Audio', color: 'orange', icon: WaveformIcon },
  tabular: { label: 'Tabular', color: 'grape', icon: TableIcon },
}

export function ProjectPage() {
  const { projectId } = routeApi.useParams()
  const navigate = useNavigate()
  const eden = useEden()
  const queryClient = useQueryClient()

  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const { data: runsData } = useTrainingRuns(projectId)
  const trainingRuns = runsData?.runs ?? []

  const dataset = activeProject.dataset
  const taskDescriptor = getTaskDescriptor(activeProject.task)
  const modality = (dataset?.modality ?? taskDescriptor?.modality ?? 'vision') as DatasetModality
  const modalityMeta = MODALITY_CONFIG[modality] ?? MODALITY_CONFIG.vision

  const draftItemsCount = dataset?.draft?.itemCount ?? 0
  const snapshots = dataset?.versions ?? []
  const readySnapshotsCount = snapshots.filter((v) => v.status === 'ready').length
  const classesCount = dataset?.classes?.length ?? 0
  const totalRunsCount = activeProject.runCount ?? trainingRuns.length ?? 0
  const hasReadySnapshot = readySnapshotsCount > 0

  // Derive workflow steps dynamically based on task type and state
  const workflowSteps = useMemo(() => {
    const isClassification = isClassificationTask(activeProject.task)

    const steps = [
      {
        id: 'data',
        label: 'Upload',
        shortDesc: 'Ingest raw items and build the dataset pool',
        desc: 'Upload images, raw text lines, audio clips, or CSV records to build your training pool.',
        icon: DatabaseIcon,
        to: '/project/$projectId/upload' as
          | '/project/$projectId/upload'
          | '/project/$projectId/classes'
          | '/project/$projectId/dataset'
          | '/project/$projectId/training',
        color: 'primary',
        statusLabel: `${draftItemsCount} items in pool`,
        isComplete: draftItemsCount > 0,
        disabled: false,
        actionText: 'Upload Items',
      },
    ]

    if (isClassification) {
      steps.push({
        id: 'labeling',
        label: 'Classes',
        shortDesc: 'Define label categories for annotation',
        desc: 'Create label classes, then assign them to items from the Dataset page.',
        icon: ListChecksIcon,
        to: '/project/$projectId/classes' as const,
        color: 'violet',
        statusLabel: `${classesCount} classes configured`,
        isComplete: classesCount > 0,
        disabled: false,
        actionText: 'Configure Classes',
      })
    }

    steps.push(
      {
        id: 'dataset',
        label: 'Dataset Versions & Splits',
        shortDesc: 'Assign splits and lock version snapshots',
        desc: 'Assign train, validation, and test splits, and lock immutable snapshots for reproducible training.',
        icon: StackIcon,
        to: '/project/$projectId/dataset' as const,
        color: 'blue',
        statusLabel: readySnapshotsCount > 0 ? `${readySnapshotsCount} snapshot ready` : 'Draft version ready',
        isComplete: readySnapshotsCount > 0,
        disabled: false,
        actionText: 'Create Snapshot',
      },
      {
        id: 'training',
        label: 'Model Training',
        shortDesc: 'Ludwig AutoML neural network training',
        desc: 'Train deep learning architectures on dataset snapshots and track real-time loss and accuracy curves.',
        icon: BrainIcon,
        to: '/project/$projectId/training' as const,
        color: 'teal',
        statusLabel:
          totalRunsCount > 0
            ? `${totalRunsCount} runs executed`
            : hasReadySnapshot
              ? 'Ready to train'
              : 'Snapshot required',
        isComplete: totalRunsCount > 0,
        disabled: !hasReadySnapshot,
        actionText: 'Launch Training',
      },
      // {
      //   id: 'models',
      //   label: 'Models & Checkpoints',
      //   shortDesc: 'Evaluation metrics & weight exports',
      //   desc: 'Review model accuracy, compare evaluation checkpoints, and export trained PyTorch weights.',
      //   icon: PackageIcon,
      //   to: '/project/$projectId/models' as const,
      //   color: 'indigo',
      //   statusLabel: totalRunsCount > 0 ? 'Checkpoints available' : 'Requires completed run',
      //   isComplete: totalRunsCount > 0,
      //   disabled: totalRunsCount === 0,
      //   actionText: 'Inspect Models',
      // },
      // {
      //   id: 'inference',
      //   label: 'Inference & Live Testing',
      //   shortDesc: 'Real-time prediction & evaluation',
      //   desc: 'Test your trained models in real-time with sample inputs or batch evaluation.',
      //   icon: CrosshairIcon,
      //   to: '/project/$projectId/inference' as const,
      //   color: 'orange',
      //   statusLabel: totalRunsCount > 0 ? 'Ready for testing' : 'Requires trained model',
      //   isComplete: totalRunsCount > 0,
      //   disabled: totalRunsCount === 0,
      //   actionText: 'Test Predictions',
      // },
    )

    return steps
  }, [
    activeProject.task,
    draftItemsCount,
    classesCount,
    readySnapshotsCount,
    totalRunsCount,
    hasReadySnapshot,
  ])

  const handleEditClick = () => {
    modals.open({
      title: 'Edit project',
      centered: true,
      children: (
        <UpdateProjectModal
          projectId={activeProject.id}
          projectName={activeProject.name}
          projectDescription={activeProject.description}
        />
      ),
    })
  }

  const deleteProject = useMutation({
    mutationFn: async () => {
      const { error } = await rest.projects({ projectId }).delete()
      if (error) throw error
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects.get.queryKey() })
      notifications.show({
        title: 'Project deleted',
        message: `"${activeProject.name}" has been removed`,
        color: 'green',
      })
      navigate({ to: '/' })
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to delete project', color: 'red' })
    },
  })

  const handleDeleteClick = () => {
    confirmDelete({
      title: 'Delete project',
      message: (
        <>
          Are you sure you want to delete <strong>{activeProject.name}</strong>? This permanently removes its dataset,
          versions, items, training runs, and exports. This cannot be undone.
        </>
      ),
      onConfirm: () => deleteProject.mutate(),
    })
  }

  return (
    <Box>
      <Stack gap="lg">
        <Grid gap="lg" align="flex-start">
          <Grid.Col span={{ base: 12, lg: 5 }}>
            <Stack gap="lg">
              <Paper withBorder p="lg" radius="md">
                <Stack gap="md">
                  <Group justify="space-between">
                    <Title order={3}>{activeProject.name}</Title>
                    <Group justify="space-between" align="center">
                      <Group gap="xs">
                        <Button
                          variant="default"
                          size="xs"
                          leftSection={<PencilSimpleIcon size={14} />}
                          onClick={handleEditClick}
                        >
                          Edit Details
                        </Button>
                        <Button
                          variant="subtle"
                          color="red"
                          size="xs"
                          leftSection={<TrashIcon size={14} />}
                          onClick={handleDeleteClick}
                          loading={deleteProject.isPending}
                        >
                          Delete
                        </Button>
                      </Group>
                    </Group>
                  </Group>

                  <Text size="sm" c="dimmed">
                    {activeProject.description || 'No description provided for this project.'}
                  </Text>
                </Stack>
              </Paper>

              {/* Quick Stat Summary Tiles */}
              <SimpleGrid cols={2} spacing="sm">
                <StatCard compact icon={DatabaseIcon} color="primary" label="Pool Items" value={draftItemsCount} />
                <StatCard
                  compact
                  icon={ListChecksIcon}
                  color="violet"
                  label={isClassificationTask(activeProject.task) ? 'Classes' : 'Target'}
                  value={isClassificationTask(activeProject.task) ? classesCount : 'Single'}
                />
                <StatCard compact icon={StackIcon} color="blue" label="Snapshots" value={readySnapshotsCount} />
                <StatCard compact icon={BrainIcon} color="teal" label="Runs" value={totalRunsCount} />
              </SimpleGrid>

              {/* ML Task & Architecture Specification Card */}
              <Paper withBorder p="lg" radius="md">
                <Group gap="xs" mb="md">
                  <ThemeIcon size="md" variant="light" color="primary" radius="md">
                    <CpuIcon size={18} />
                  </ThemeIcon>
                  <Title order={4}>ML Task Specification</Title>
                </Group>

                <Stack gap="xs">
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed" fw={600} tt="uppercase">
                      Task Name
                    </Text>
                    <Text size="sm" fw={600}>
                      {taskDescriptor?.label ?? activeProject.task}
                    </Text>
                  </Group>
                  <Divider />

                  <Group justify="space-between">
                    <Text size="xs" c="dimmed" fw={600} tt="uppercase">
                      Modality
                    </Text>
                    <Badge variant="light" color={modalityMeta.color} size="sm">
                      {modalityMeta.label}
                    </Badge>
                  </Group>
                  <Divider />

                  <Group justify="space-between">
                    <Text size="xs" c="dimmed" fw={600} tt="uppercase">
                      Input Payload
                    </Text>
                    <Badge variant="outline" size="sm">
                      {taskDescriptor?.itemSpec.payload ?? 'file'}
                    </Badge>
                  </Group>
                  <Divider />

                  <Group justify="space-between">
                    <Text size="xs" c="dimmed" fw={600} tt="uppercase">
                      Annotation Type
                    </Text>
                    <Text size="sm" fw={500}>
                      {taskDescriptor?.annotation.type ?? 'classification'}
                    </Text>
                  </Group>
                </Stack>
              </Paper>
            </Stack>
          </Grid.Col>

          <Grid.Col span={{ base: 12, lg: 7 }}>
            <Stack gap="md">
              <Paper withBorder p="md" radius="md">
                <Group justify="space-between" align="center">
                  <div>
                    <Title order={3}>ML Pipeline Lifecycle</Title>
                    <Text size="sm" c="dimmed">
                      Sequential workflow from raw data ingestion to live inference
                    </Text>
                  </div>
                  <Badge variant="light" color="primary" size="md">
                    {workflowSteps.filter((s) => s.isComplete).length} / {workflowSteps.length} Stages Ready
                  </Badge>
                </Group>
              </Paper>

              {/* Vertical Stepper Cards */}
              <Stack gap="md">
                {workflowSteps.map((step, idx) => {
                  const StepIcon = step.icon
                  return (
                    <Card
                      key={step.id}
                      withBorder
                      padding="md"
                      radius="md"
                      className="card-elevated"
                      style={{
                        cursor: step.disabled ? 'not-allowed' : 'pointer',
                        opacity: step.disabled ? 0.6 : 1,
                        transition: 'all 150ms ease',
                      }}
                      onClick={() => !step.disabled && navigate({ to: step.to, params: { projectId } })}
                    >
                      <Group justify="space-between" align="flex-start" wrap="nowrap">
                        <Group gap="md" align="flex-start" wrap="nowrap" style={{ flexGrow: 1 }}>
                          <ThemeIcon
                            size={42}
                            variant={step.isComplete ? 'filled' : 'light'}
                            color={step.color}
                            radius="md"
                          >
                            <StepIcon size={22} />
                          </ThemeIcon>

                          <div style={{ flexGrow: 1 }}>
                            <Group justify="space-between" align="center">
                              <Group gap="xs">
                                <Text size="xs" fw={700} c={step.color} tt="uppercase">
                                  Step {idx + 1}
                                </Text>
                                <Text fw={600} size="md">
                                  {step.label}
                                </Text>
                              </Group>

                              {step.isComplete && (
                                <Group gap={4}>
                                  <CheckCircleIcon size={16} color="var(--mantine-color-teal-filled)" />
                                  <Text size="xs" c="teal" fw={600}>
                                    Ready
                                  </Text>
                                </Group>
                              )}
                            </Group>

                            <Text size="sm" c="dimmed" mt={2}>
                              {step.desc}
                            </Text>

                            <Group justify="space-between" align="center" mt="sm">
                              <Badge
                                variant="light"
                                color={step.isComplete ? 'teal' : step.disabled ? 'gray' : step.color}
                                size="sm"
                              >
                                {step.statusLabel}
                              </Badge>

                              <Button
                                size="xs"
                                variant="light"
                                color={step.color}
                                rightSection={<ArrowRightIcon size={14} />}
                                disabled={step.disabled}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  navigate({ to: step.to, params: { projectId } })
                                }}
                              >
                                {step.actionText}
                              </Button>
                            </Group>
                          </div>
                        </Group>
                      </Group>
                    </Card>
                  )
                })}
              </Stack>

              {/* Recent Training Runs Activity Card */}
              <Paper withBorder p="lg" radius="md" mt="xs">
                <Group justify="space-between" align="center" mb="sm">
                  <Group gap="xs">
                    <ThemeIcon size="md" variant="light" color="teal" radius="md">
                      <BrainIcon size={18} />
                    </ThemeIcon>
                    <Title order={4}>Recent Training Activity</Title>
                  </Group>
                  {trainingRuns.length > 0 && (
                    <Button
                      variant="subtle"
                      size="xs"
                      rightSection={<ArrowRightIcon size={12} />}
                      onClick={() => navigate({ to: '/project/$projectId/training', params: { projectId } })}
                    >
                      View all ({trainingRuns.length})
                    </Button>
                  )}
                </Group>

                {trainingRuns.length === 0 ? (
                  <Stack align="center" justify="center" py="md" gap="xs">
                    <Text size="sm" c="dimmed" ta="center">
                      No training runs executed yet.
                    </Text>
                    <Text size="xs" c="dimmed" ta="center" maw={360}>
                      Create a dataset snapshot, then launch a training run to train neural network weights with Ludwig.
                    </Text>
                    {hasReadySnapshot && (
                      <Button
                        size="xs"
                        variant="light"
                        color="teal"
                        leftSection={<PlusIcon size={14} />}
                        onClick={() => navigate({ to: '/project/$projectId/training', params: { projectId } })}
                        mt="xs"
                      >
                        Start Training Run
                      </Button>
                    )}
                  </Stack>
                ) : (
                  <Table highlightOnHover verticalSpacing="xs">
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Run Name</Table.Th>
                        <Table.Th>Status</Table.Th>
                        <Table.Th>Started</Table.Th>
                        <Table.Th style={{ textAlign: 'right' }}>Action</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {trainingRuns.slice(0, 3).map((run) => (
                        <Table.Tr key={run.id}>
                          <Table.Td>
                            <Text size="sm" fw={600} truncate lineClamp={1}>
                              {run.name}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            <Badge variant="light" color={STATUS_COLORS[run.status] ?? 'gray'} size="xs">
                              {run.status}
                            </Badge>
                          </Table.Td>
                          <Table.Td>
                            <Text size="xs" c="dimmed">
                              {formatDate(run.createdAt)}
                            </Text>
                          </Table.Td>
                          <Table.Td style={{ textAlign: 'right' }}>
                            <Button
                              size="compact-xs"
                              variant="light"
                              onClick={() =>
                                navigate({
                                  to: '/project/$projectId/training',
                                  params: { projectId },
                                  search: { runId: run.id },
                                })
                              }
                            >
                              View
                            </Button>
                          </Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                )}
              </Paper>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>
    </Box>
  )
}

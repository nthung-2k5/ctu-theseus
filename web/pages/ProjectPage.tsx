import { Alert, Badge, Button, Group, Paper, SimpleGrid, Stack, Table, Text } from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  ArchiveIcon,
  ArrowRightIcon,
  BrainIcon,
  CheckCircleIcon,
  CircleIcon,
  type DatabaseIcon,
  ExportIcon,
  PencilSimpleIcon,
  StackIcon,
  TagIcon,
  TrashIcon,
  UploadSimpleIcon,
  WarningCircleIcon,
} from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { UpdateProjectModal } from '@public/components/UpdateProjectModal'
import { ClassDot, confirmDelete, PageHeader, SectionLabel, StatCard, StatusBadge } from '@public/components/ui'
import type { DatasetModality } from '@public/lib/api/enums'
import { deleteProject as deleteProjectRequest } from '@public/lib/api/generated/projects/projects'
import { formatDate } from '@public/lib/format'
import { MODALITY_META } from '@public/lib/modality'
import { invalidateProjectList, projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import { getTaskDescriptor, isClassificationTask } from '@public/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, useNavigate } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/')

type StepTarget =
  | '/project/$projectId/upload'
  | '/project/$projectId/classes'
  | '/project/$projectId/dataset'
  | '/project/$projectId/snapshots'
  | '/project/$projectId/experiments'
  | '/project/$projectId/export'

interface Step {
  id: string
  label: string
  desc: string
  icon: typeof DatabaseIcon
  to: StepTarget
  status: string
  done: boolean
  disabled: boolean
  action: string
}

export function ProjectPage() {
  const { projectId } = routeApi.useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const { data: runsData } = useTrainingRuns(projectId)
  const runs = runsData?.runs ?? []

  const dataset = project.dataset
  const descriptor = getTaskDescriptor(project.task)
  const modality = (dataset?.modality ?? descriptor?.modality ?? 'vision') as DatasetModality
  const modalityMeta = MODALITY_META[modality] ?? MODALITY_META.vision
  const classification = isClassificationTask(project.task)

  const poolItems = dataset?.draft?.itemCount ?? 0
  const classes = dataset?.classes ?? []
  const snapshots = dataset?.versions ?? []
  const readySnapshots = snapshots.filter((v) => v.status === 'ready').length
  const runCount = project.runCount ?? runs.length
  const succeededRuns = runs.filter((r) => r.status === 'succeeded').length
  const hasReadySnapshot = readySnapshots > 0

  const steps: Step[] = [
    {
      id: 'upload',
      label: 'Upload',
      desc: 'Add images, audio, text or CSV rows to the pool.',
      icon: UploadSimpleIcon,
      to: '/project/$projectId/upload',
      status: `${poolItems} items in pool`,
      done: poolItems > 0,
      disabled: false,
      action: 'Upload items',
    },
    ...(classification
      ? [
          {
            id: 'classes',
            label: 'Classes',
            desc: 'Define the label classes, then assign them on the Dataset page.',
            icon: TagIcon,
            to: '/project/$projectId/classes' as const,
            status: `${classes.length} classes`,
            done: classes.length > 0,
            disabled: false,
            action: 'Configure classes',
          },
        ]
      : []),
    {
      id: 'dataset',
      label: 'Dataset',
      desc: 'Browse the draft, assign splits and labels.',
      icon: StackIcon,
      to: '/project/$projectId/dataset',
      status: poolItems > 0 ? `${poolItems} items` : 'Empty draft',
      done: poolItems > 0,
      disabled: false,
      action: 'Open dataset',
    },
    {
      id: 'snapshots',
      label: 'Snapshot',
      desc: 'Freeze the draft, optionally augmented, as an immutable version.',
      icon: ArchiveIcon,
      to: '/project/$projectId/snapshots',
      status: hasReadySnapshot ? `${readySnapshots} ready` : 'None yet',
      done: hasReadySnapshot,
      disabled: poolItems === 0,
      action: 'Open snapshots',
    },
    {
      id: 'experiments',
      label: 'Train',
      desc: 'Launch runs or sweeps and watch metrics stream live.',
      icon: BrainIcon,
      to: '/project/$projectId/experiments',
      status:
        runCount > 0
          ? `${runCount} run${runCount === 1 ? '' : 's'}`
          : hasReadySnapshot
            ? 'Ready to train'
            : 'Needs a snapshot',
      done: runCount > 0,
      disabled: !hasReadySnapshot,
      action: 'Open experiments',
    },
    {
      id: 'export',
      label: 'Test & export',
      desc: 'Try a trained model, then package it.',
      icon: ExportIcon,
      to: '/project/$projectId/export',
      status:
        succeededRuns > 0 ? `${succeededRuns} model${succeededRuns === 1 ? '' : 's'} ready` : 'Needs a finished run',
      done: succeededRuns > 0,
      disabled: succeededRuns === 0,
      action: 'Open export',
    },
  ]

  const handleEdit = () =>
    modals.open({
      title: 'Edit project',
      centered: true,
      children: (
        <UpdateProjectModal
          projectId={project.id}
          projectName={project.name}
          projectDescription={project.description}
        />
      ),
    })

  const deleteProject = useMutation({
    mutationFn: async () => {
      await deleteProjectRequest(projectId)
    },
    onSuccess: () => {
      invalidateProjectList(queryClient)
      notifications.show({ title: 'Project deleted', message: `"${project.name}" has been removed`, color: 'green' })
      navigate({ to: '/projects' })
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to delete project', color: 'red' }),
  })

  const handleDelete = () =>
    confirmDelete({
      title: 'Delete project',
      message: (
        <>
          Are you sure you want to delete <strong>{project.name}</strong>? This permanently removes its dataset,
          versions, items, training runs and exports. This cannot be undone.
        </>
      ),
      onConfirm: () => deleteProject.mutate(),
    })

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title={project.name}
        description={project.description || 'No description'}
        badges={
          <>
            <Badge color={modalityMeta.color}>{modalityMeta.label}</Badge>
            <Badge variant="outline" color="gray">
              {descriptor?.label ?? project.task}
            </Badge>
          </>
        }
        actions={
          <Button variant="default" leftSection={<PencilSimpleIcon size={14} />} onClick={handleEdit}>
            Edit details
          </Button>
        }
      />

      {classification && classes.length < 2 && (
        <Alert color="yellow" p="xs" icon={<WarningCircleIcon size={16} />} title="Not enough classes">
          A classification project needs at least two classes before it can be trained on.
        </Alert>
      )}

      <SimpleGrid cols={{ base: 2, md: 4 }} spacing="sm">
        <StatCard label="Pool items" value={poolItems} />
        <StatCard label={classification ? 'Classes' : 'Target'} value={classification ? classes.length : 'Single'} />
        <StatCard
          label="Snapshots"
          value={readySnapshots}
          hint={snapshots.length > readySnapshots ? `${snapshots.length} total` : undefined}
        />
        <StatCard label="Runs" value={runCount} hint={succeededRuns > 0 ? `${succeededRuns} succeeded` : undefined} />
      </SimpleGrid>

      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="sm">
        <Paper p="md">
          <Stack gap="xs">
            <Group justify="space-between">
              <SectionLabel>Get going</SectionLabel>
              <Text size="xs" c="dimmed" className="tnum">
                {steps.filter((s) => s.done).length} / {steps.length} stages ready
              </Text>
            </Group>
            {steps.map((step) => {
              const StepIcon = step.icon
              return (
                <Group key={step.id} justify="space-between" wrap="nowrap" gap="sm">
                  <Group gap="sm" wrap="nowrap" style={{ minWidth: 0 }}>
                    {step.done ? (
                      <CheckCircleIcon size={18} weight="fill" color="var(--mantine-color-teal-5)" />
                    ) : (
                      <CircleIcon size={18} color="var(--mantine-color-dimmed)" />
                    )}
                    <div style={{ minWidth: 0 }}>
                      <Group gap={6} wrap="nowrap">
                        <StepIcon size={14} />
                        <Text size="sm" fw={500}>
                          {step.label}
                        </Text>
                        <Badge color={step.done ? 'teal' : 'gray'}>{step.status}</Badge>
                      </Group>
                      <Text size="xs" c="dimmed" truncate>
                        {step.desc}
                      </Text>
                    </div>
                  </Group>
                  <Button
                    variant="light"
                    size="compact-sm"
                    rightSection={<ArrowRightIcon size={12} />}
                    disabled={step.disabled}
                    onClick={() => navigate({ to: step.to, params: { projectId } })}
                  >
                    {step.action}
                  </Button>
                </Group>
              )
            })}
            {classification && classes.length > 0 && (
              <Group gap={6} mt="xs">
                {classes.slice(0, 12).map((c) => (
                  <Group key={c.classId} gap={4} wrap="nowrap">
                    <ClassDot color={c.uiColorHex} />
                    <Text size="xs">{c.name}</Text>
                  </Group>
                ))}
                {classes.length > 12 && (
                  <Text size="xs" c="dimmed">
                    +{classes.length - 12}
                  </Text>
                )}
              </Group>
            )}
          </Stack>
        </Paper>

        <Paper p="md">
          <Stack gap="xs">
            <Group justify="space-between">
              <SectionLabel>Recent experiments</SectionLabel>
              {runs.length > 0 && (
                <Button
                  variant="subtle"
                  size="compact-xs"
                  rightSection={<ArrowRightIcon size={12} />}
                  onClick={() => navigate({ to: '/project/$projectId/experiments', params: { projectId } })}
                >
                  View all ({runs.length})
                </Button>
              )}
            </Group>
            {runs.length === 0 ? (
              <Text size="sm" c="dimmed" ta="center" py="md">
                No training runs yet. Build a snapshot, then launch a run.
              </Text>
            ) : (
              <Table highlightOnHover verticalSpacing={6}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Run</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th>Started</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {runs.slice(0, 5).map((run) => (
                    <Table.Tr
                      key={run.id}
                      style={{ cursor: 'pointer' }}
                      onClick={() =>
                        navigate({
                          to: '/project/$projectId/experiments/$runId',
                          params: { projectId, runId: run.id },
                        })
                      }
                    >
                      <Table.Td>
                        <Text size="sm" fw={500} truncate>
                          {run.name}
                        </Text>
                      </Table.Td>
                      <Table.Td>
                        <StatusBadge value={run.status} colorMap={STATUS_COLORS} />
                      </Table.Td>
                      <Table.Td>
                        <Text size="xs" c="dimmed" className="tnum">
                          {formatDate(run.createdAt)}
                        </Text>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            )}
          </Stack>
        </Paper>
      </SimpleGrid>

      <Paper p="md" style={{ borderColor: 'var(--mantine-color-red-8)' }}>
        <Group justify="space-between" wrap="nowrap">
          <div>
            <Text size="sm" fw={500}>
              Delete this project
            </Text>
            <Text size="xs" c="dimmed">
              Permanently removes its dataset, snapshots, items, runs and exports.
            </Text>
          </div>
          <Button
            color="red"
            variant="light"
            leftSection={<TrashIcon size={14} />}
            loading={deleteProject.isPending}
            onClick={handleDelete}
          >
            Delete project
          </Button>
        </Group>
      </Paper>
    </div>
  )
}

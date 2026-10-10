import { ActionIcon, Alert, Badge, Button, Group, Menu, Paper, SimpleGrid, Stack, Text } from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import {
  ArchiveIcon,
  BrainIcon,
  CheckCircleIcon,
  DatabaseIcon,
  DotsThreeIcon,
  PencilSimpleIcon,
  PlayIcon,
  TrashIcon,
  TrophyIcon,
  WarningCircleIcon,
} from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { UpdateProjectModal } from '@public/components/UpdateProjectModal'
import { confirmDelete, LinkButton, PageHeader, SectionLabel, StatusBadge } from '@public/components/ui'
import type { DatasetModality } from '@public/lib/api/enums'
import { deleteProject as deleteProjectRequest } from '@public/lib/api/generated/projects/projects'
import { formatDate } from '@public/lib/format'
import { MODALITY_META } from '@public/lib/modality'
import { invalidateProjectList, projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import { bestRun, scoreOf } from '@public/lib/runScore'
import { getTaskDescriptor, isClassificationTask } from '@public/lib/tasks'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, useNavigate } from '@tanstack/react-router'
import type { ElementType, ReactNode } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/')

interface Step {
  id: string
  label: string
  icon: ElementType
  /** The one number worth reading at a glance. */
  headline: string
  detail: string
  done: boolean
  /** Can't be started yet: an earlier step has to finish first. */
  blocked: boolean
  /** What the disabled button says while blocked. */
  blockedLabel: string
  cta: (variant: 'filled' | 'light') => ReactNode
}

const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`

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
  const hasReadySnapshot = readySnapshots > 0
  const runCount = project.runCount ?? runs.length
  const succeeded = runs.filter((r) => r.status === 'succeeded')
  const activeRuns = runs.filter((r) => r.status === 'running' || r.status === 'queued').length
  const needsClasses = classification && classes.length < 2

  const best = bestRun(succeeded)
  const latest = runs[0] ?? null // the list is newest first
  // The model to try: the best-scoring one, else the most recent that finished.
  const tryRun = best ?? succeeded[0] ?? null

  const steps: Step[] = [
    {
      id: 'data',
      label: 'Data',
      icon: DatabaseIcon,
      headline: plural(poolItems, 'item'),
      detail: classification ? `${classes.length} ${classes.length === 1 ? 'class' : 'classes'}` : 'In the draft',
      done: poolItems > 0 && !needsClasses,
      blocked: false,
      blockedLabel: 'Add data',
      cta: (variant) =>
        poolItems === 0 ? (
          <LinkButton
            to="/project/$projectId/data"
            params={{ projectId }}
            search={{ page: 1, perPage: 20, add: true }}
            variant={variant}
          >
            Add data
          </LinkButton>
        ) : needsClasses ? (
          <LinkButton
            to="/project/$projectId/data"
            params={{ projectId }}
            search={{ page: 1, perPage: 20, tab: 'classes' }}
            variant={variant}
          >
            Define classes
          </LinkButton>
        ) : (
          <LinkButton
            to="/project/$projectId/data"
            params={{ projectId }}
            search={{ page: 1, perPage: 20 }}
            variant={variant}
          >
            Open data
          </LinkButton>
        ),
    },
    {
      id: 'snapshot',
      label: 'Snapshot',
      icon: ArchiveIcon,
      headline: `${readySnapshots.toLocaleString()} ready`,
      detail: snapshots.length > readySnapshots ? `${snapshots.length} in total` : 'Freeze the draft as a version',
      done: hasReadySnapshot,
      blocked: poolItems === 0,
      blockedLabel: 'Create snapshot',
      cta: (variant) =>
        hasReadySnapshot ? (
          <LinkButton to="/project/$projectId/snapshots" params={{ projectId }} variant={variant}>
            Open snapshots
          </LinkButton>
        ) : (
          <LinkButton to="/project/$projectId/snapshots/new" params={{ projectId }} variant={variant}>
            Create snapshot
          </LinkButton>
        ),
    },
    {
      id: 'train',
      label: 'Train',
      icon: BrainIcon,
      headline: plural(runCount, 'run'),
      detail: activeRuns > 0 ? `${activeRuns} in progress` : 'Runs and sweeps',
      done: succeeded.length > 0,
      blocked: !hasReadySnapshot,
      blockedLabel: 'New run',
      cta: (variant) =>
        runCount > 0 ? (
          <LinkButton to="/project/$projectId/training" params={{ projectId }} variant={variant}>
            Open training
          </LinkButton>
        ) : (
          <LinkButton
            to="/project/$projectId/training/new"
            params={{ projectId }}
            search={{ mode: 'run' }}
            variant={variant}
          >
            New run
          </LinkButton>
        ),
    },
    {
      id: 'use',
      label: 'Try & export',
      icon: PlayIcon,
      headline: succeeded.length > 0 ? plural(succeeded.length, 'model') : 'No model yet',
      detail: succeeded.length > 0 ? 'Ready to test and package' : 'Needs a finished run',
      // Using a model is never "finished"; this step is the end of the road, not a box to tick.
      done: false,
      blocked: !tryRun,
      blockedLabel: 'Try a model',
      cta: (variant) =>
        tryRun ? (
          <LinkButton
            to="/project/$projectId/training/$runId/try"
            params={{ projectId, runId: tryRun.id }}
            variant={variant}
          >
            Try {best ? 'best' : 'latest'} model
          </LinkButton>
        ) : null,
    },
  ]
  const nextId = steps.find((step) => !step.done && !step.blocked)?.id

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
          <>
            <Button variant="default" leftSection={<PencilSimpleIcon size={14} />} onClick={handleEdit}>
              Edit details
            </Button>
            <Menu position="bottom-end" width={200}>
              <Menu.Target>
                <ActionIcon
                  variant="default"
                  size="lg"
                  aria-label="More project actions"
                  loading={deleteProject.isPending}
                >
                  <DotsThreeIcon size={16} weight="bold" />
                </ActionIcon>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Item color="red" leftSection={<TrashIcon size={14} />} onClick={handleDelete}>
                  Delete project…
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </>
        }
      />

      {needsClasses && (
        <Alert color="yellow" p="xs" icon={<WarningCircleIcon size={16} />} title="Not enough classes">
          A classification project needs at least two classes before it can be trained on.
        </Alert>
      )}

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="sm">
        {steps.map((step, index) => {
          const isNext = step.id === nextId
          const StepIcon = step.icon
          return (
            <Paper
              key={step.id}
              p="md"
              style={isNext ? { borderColor: 'var(--mantine-primary-color-filled)', borderWidth: 2 } : undefined}
              aria-current={isNext ? 'step' : undefined}
            >
              <Stack gap="xs" h="100%" justify="space-between">
                <Stack gap={6}>
                  <Group justify="space-between" wrap="nowrap">
                    <Group gap={6} wrap="nowrap">
                      <StepIcon size={16} color="var(--mantine-color-dimmed)" />
                      <SectionLabel>
                        {index + 1} · {step.label}
                      </SectionLabel>
                    </Group>
                    {step.done ? (
                      <CheckCircleIcon size={18} weight="fill" color="var(--mantine-color-teal-5)" aria-label="Done" />
                    ) : isNext ? (
                      <Badge>Next</Badge>
                    ) : null}
                  </Group>
                  <Text fw={600} size="xl" className="tnum" c={step.blocked ? 'dimmed' : undefined}>
                    {step.headline}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {step.detail}
                  </Text>
                </Stack>
                {step.blocked ? (
                  <Button variant="light" disabled>
                    {step.blockedLabel}
                  </Button>
                ) : (
                  step.cta(isNext ? 'filled' : 'light')
                )}
              </Stack>
            </Paper>
          )
        })}
      </SimpleGrid>

      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="sm">
        <RunCard
          projectId={projectId}
          title="Best model"
          icon={<TrophyIcon size={14} weight="fill" color="var(--mantine-color-yellow-6)" />}
          run={best}
          empty="No evaluated model yet. Finish a run and its score shows up here."
          actions={
            best && (
              <>
                <LinkButton
                  to="/project/$projectId/training/$runId/try"
                  params={{ projectId, runId: best.id }}
                  variant="light"
                >
                  Try
                </LinkButton>
                <LinkButton
                  to="/project/$projectId/training/$runId/export"
                  params={{ projectId, runId: best.id }}
                  variant="light"
                >
                  Export
                </LinkButton>
              </>
            )
          }
        />
        <RunCard
          projectId={projectId}
          title="Latest run"
          icon={<BrainIcon size={14} />}
          run={latest}
          empty="No training runs yet. Build a snapshot, then launch a run."
          actions={
            runs.length > 0 && (
              <LinkButton to="/project/$projectId/training" params={{ projectId }} variant="subtle">
                All runs ({runs.length})
              </LinkButton>
            )
          }
        />
      </SimpleGrid>
    </div>
  )
}

function RunCard({
  projectId,
  title,
  icon,
  run,
  empty,
  actions,
}: {
  projectId: string
  title: string
  icon: ReactNode
  run: TrainingRunSummary | null
  empty: string
  actions: ReactNode
}) {
  const score = run ? scoreOf(run) : null
  return (
    <Paper p="md">
      <Stack gap="xs">
        <Group gap={6}>
          {icon}
          <SectionLabel>{title}</SectionLabel>
        </Group>
        {run ? (
          <>
            <Group justify="space-between" wrap="nowrap" align="flex-start">
              <div style={{ minWidth: 0 }}>
                <LinkButton
                  to="/project/$projectId/training/$runId"
                  params={{ projectId, runId: run.id }}
                  variant="subtle"
                  size="compact-md"
                  px={0}
                >
                  {run.name}
                </LinkButton>
                <Text size="xs" c="dimmed" className="tnum">
                  {run.id.slice(0, 8)} · {formatDate(run.createdAt)}
                </Text>
              </div>
              <Group gap="sm" wrap="nowrap">
                {score && (
                  <div style={{ textAlign: 'right' }}>
                    <Text fw={600} size="lg" className="tnum">
                      {score.value.toFixed(3)}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {score.label}
                    </Text>
                  </div>
                )}
                <StatusBadge value={run.status} colorMap={STATUS_COLORS} />
              </Group>
            </Group>
            <Group gap="xs">{actions}</Group>
          </>
        ) : (
          <Text size="sm" c="dimmed" py="sm">
            {empty}
          </Text>
        )}
      </Stack>
    </Paper>
  )
}

import { Group, Loader, Paper, Stack, Text, ThemeIcon } from '@mantine/core'
import { STATUS_COLORS } from '@public/components/training/constants'
import { LinkButton, StatusBadge } from '@public/components/ui'
import type { ProjectSummary } from '@public/lib/api/generated/models'
import { formatRelative } from '@public/lib/format'
import { MODALITY_META } from '@public/lib/modality'
import { type NextStep, nextStep } from '@public/lib/projectsList'
import { taskRegistry } from '@public/lib/tasks'
import { Link } from '@tanstack/react-router'
import type { ReactNode } from 'react'

/**
 * One project in the projects list, as a row or as a card. Both answer the same two questions: where is this
 * project up to, and what should I do next. The whole item opens the project (the name is a link stretched over
 * it), and the next-step button sits above that so it can be pressed on its own.
 */

function describe(p: ProjectSummary) {
  const task = taskRegistry[p.task as keyof typeof taskRegistry]
  const meta = MODALITY_META[task?.modality ?? 'vision'] ?? MODALITY_META.vision
  return { taskLabel: task?.label ?? p.task, meta }
}

function ModalityTile({ p, size = 40 }: { p: ProjectSummary; size?: number }) {
  const { meta } = describe(p)
  return (
    <ThemeIcon size={size} radius="md" variant="light" color={meta.color} title={meta.label} aria-label={meta.label}>
      <meta.icon size={size * 0.55} />
    </ThemeIcon>
  )
}

function ProjectLink({ p }: { p: ProjectSummary }) {
  return (
    <Link
      to="/project/$projectId"
      params={{ projectId: p.id }}
      className="stretched-link"
      style={{ color: 'inherit', textDecoration: 'none', fontWeight: 600 }}
    >
      <Text component="span" fw={600} truncate="end" style={{ display: 'block' }}>
        {p.name}
      </Text>
    </Link>
  )
}

function NextStepButton({ p, step }: { p: ProjectSummary; step: NextStep }) {
  const props = { size: 'compact-sm', variant: 'light', style: { position: 'relative', zIndex: 1 } } as const
  const projectId = p.id
  switch (step.kind) {
    case 'run':
      return (
        <LinkButton {...props} to="/project/$projectId/training/$runId" params={{ projectId, runId: step.runId }}>
          {step.label}
        </LinkButton>
      )
    case 'data':
      return (
        <LinkButton {...props} to="/project/$projectId/data" params={{ projectId }} search={{ page: 1, perPage: 20 }}>
          {step.label}
        </LinkButton>
      )
    case 'snapshot':
      return (
        <LinkButton {...props} to="/project/$projectId/snapshots" params={{ projectId }}>
          {step.label}
        </LinkButton>
      )
    case 'train':
      return (
        <LinkButton {...props} to="/project/$projectId/training/new" params={{ projectId }}>
          {step.label}
        </LinkButton>
      )
    default:
      return (
        <LinkButton {...props} to="/project/$projectId/training" params={{ projectId }}>
          {step.label}
        </LinkButton>
      )
  }
}

/** The run state: a live badge while something is training, else how the last run ended. */
function RunState({ p }: { p: ProjectSummary }) {
  if (p.activeRun) {
    return (
      <Group gap={4} wrap="nowrap">
        <Loader size={10} />
        <StatusBadge value={p.activeRun.status} colorMap={STATUS_COLORS} />
      </Group>
    )
  }
  return p.lastRunStatus ? (
    <StatusBadge value={p.lastRunStatus} colorMap={STATUS_COLORS} />
  ) : (
    <Text size="xs" c="dimmed">
      No runs yet
    </Text>
  )
}

function Stat({ label, children, w }: { label: string; children: ReactNode; w?: number }) {
  return (
    <div style={{ width: w, flexShrink: 0 }}>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text size="sm" fw={500} className="tnum" truncate>
        {children}
      </Text>
    </div>
  )
}

const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`

export function ProjectRow({ p }: { p: ProjectSummary }) {
  const { taskLabel } = describe(p)
  const step = nextStep(p)
  return (
    <Paper p="sm" className="project-item">
      <Group wrap="nowrap" gap="md" align="center">
        <ModalityTile p={p} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <Group gap="xs" wrap="nowrap">
            <ProjectLink p={p} />
            <Text size="xs" c="dimmed" truncate visibleFrom="sm" style={{ flexShrink: 0 }}>
              {taskLabel}
            </Text>
          </Group>
          <Text size="xs" c="dimmed" truncate>
            {p.description || 'No description'}
          </Text>
          <Text size="xs" c="dimmed" className="tnum" hiddenFrom="md">
            {plural(p.itemCount ?? 0, 'item')} · {plural(p.snapshotCount ?? 0, 'snapshot')} ·{' '}
            {plural(p.runCount ?? 0, 'run')}
          </Text>
        </div>

        <Group gap="lg" wrap="nowrap" visibleFrom="md">
          <Stat label="Items" w={64}>
            {(p.itemCount ?? 0).toLocaleString()}
          </Stat>
          <Stat label="Snapshots" w={64}>
            {p.snapshotCount ?? 0}
          </Stat>
          <Stat label="Runs" w={40}>
            {p.runCount ?? 0}
          </Stat>
          <Stat label={p.bestScore?.label ?? 'Best score'} w={84}>
            {p.bestScore ? p.bestScore.value.toFixed(3) : '—'}
          </Stat>
        </Group>

        <Stack gap={2} align="flex-end" w={104} style={{ flexShrink: 0 }} visibleFrom="sm">
          <RunState p={p} />
          <Text size="xs" c="dimmed" className="tnum">
            {formatRelative(p.lastActivityAt ?? p.createdAt)}
          </Text>
        </Stack>

        <NextStepButton p={p} step={step} />
      </Group>
    </Paper>
  )
}

export function ProjectCard({ p }: { p: ProjectSummary }) {
  const { taskLabel } = describe(p)
  const step = nextStep(p)
  return (
    <Paper p="md" className="project-item" h="100%">
      <Stack gap="sm" h="100%" justify="space-between">
        <Stack gap="sm">
          <Group wrap="nowrap" gap="sm">
            <ModalityTile p={p} size={36} />
            <div style={{ minWidth: 0 }}>
              <ProjectLink p={p} />
              <Text size="xs" c="dimmed" truncate>
                {taskLabel}
              </Text>
            </div>
          </Group>
          <Text size="xs" c="dimmed" lineClamp={2} mih={32}>
            {p.description || 'No description'}
          </Text>
          <Group gap="lg" wrap="nowrap">
            <Stat label="Items">{(p.itemCount ?? 0).toLocaleString()}</Stat>
            <Stat label="Snapshots">{p.snapshotCount ?? 0}</Stat>
            <Stat label="Runs">{p.runCount ?? 0}</Stat>
            <Stat label={p.bestScore?.label ?? 'Best score'}>{p.bestScore ? p.bestScore.value.toFixed(3) : '—'}</Stat>
          </Group>
        </Stack>
        <Group justify="space-between" wrap="nowrap">
          <Stack gap={2}>
            <RunState p={p} />
            <Text size="xs" c="dimmed" className="tnum">
              {formatRelative(p.lastActivityAt ?? p.createdAt)}
            </Text>
          </Stack>
          <NextStepButton p={p} step={step} />
        </Group>
      </Stack>
    </Paper>
  )
}

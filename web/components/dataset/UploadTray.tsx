/**
 * The upload tray: every running upload, and the result of every finished one, pinned to a corner of the app so a
 * batch carries on while the user closes the drawer or moves to another page.
 *
 * A running job shows progress, the time left and Cancel. A finished one is a result card: what was added, what was
 * already there, what failed (with a retry that sends only those), how the classes stand now and what to do next.
 * Jobs live in memory, so a page reload ends them; the tray asks the browser to confirm before that happens.
 */

import {
  ActionIcon,
  Badge,
  Button,
  Collapse,
  Group,
  Paper,
  Progress,
  ScrollArea,
  Stack,
  Text,
  Tooltip,
} from '@mantine/core'
import { modals } from '@mantine/modals'
import { CheckCircleIcon, WarningCircleIcon, XIcon } from '@phosphor-icons/react'
import { LinkButton, ProportionBar } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import { projectDetailQueryOptions, useClassStats, useLabelClasses } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import {
  cancelJob,
  dismissJob,
  hasRunningJob,
  retryJob,
  secondsLeft,
  type UploadJobView,
} from '@public/lib/upload/jobs'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { plural } from './useCommit'
import { useUploadJobs } from './useUploadJobs'

/** "45 s", "3 min", "1 h 5 min". */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min`
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

/** The current time, ticking once a second while `active`. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [active])
  return now
}

function RunningJob({ job }: { job: UploadJobView }) {
  const now = useNow(true)
  const left = secondsLeft(job, now)
  return (
    <Stack gap={6}>
      <Group justify="space-between" wrap="nowrap">
        <Text size="sm" fw={600} truncate>
          Uploading {job.label}
        </Text>
        <Button size="compact-xs" variant="subtle" color="red" onClick={() => cancelJob(job.id)}>
          Cancel
        </Button>
      </Group>
      <Progress value={job.fraction * 100} animated size="sm" />
      <Group justify="space-between" wrap="nowrap">
        <Text size="xs" c="dimmed" truncate>
          {job.phase ?? `${Math.round(job.fraction * 100)}% · ${job.uploaded.toLocaleString()} added`}
          {job.failed > 0 ? ` · ${job.failed.toLocaleString()} failed` : ''}
        </Text>
        {left !== null && (
          <Text size="xs" c="dimmed" style={{ flexShrink: 0 }}>
            about {formatDuration(left)} left
          </Text>
        )}
      </Group>
    </Stack>
  )
}

/** How the classes stand after the upload: the whole draft, since that is what a snapshot would freeze. */
function ClassBalance({ projectId }: { projectId: string }) {
  const stats = useClassStats(projectId)
  const classes = useLabelClasses(projectId)
  const names = new Map((classes.data?.classes ?? []).map((c) => [c.classId, c.name]))
  const rows = (stats.data?.stats ?? []).filter((s) => s.total > 0).sort((a, b) => b.total - a.total)
  if (rows.length < 2) return null
  const smallest = rows[rows.length - 1]
  const minimum = stats.data?.minItemsPerClass ?? 3
  const nameOf = (id: string) => names.get(id) ?? 'A class'
  return (
    <Text size="xs" c="dimmed">
      Largest class: {nameOf(rows[0].classId)} ({rows[0].total.toLocaleString()}). Smallest: {nameOf(smallest.classId)}{' '}
      ({smallest.total.toLocaleString()}){smallest.total < minimum ? ` — too few to reach every split.` : '.'}
    </Text>
  )
}

function FinishedJob({ job }: { job: UploadJobView }) {
  const { data } = useQuery(projectDetailQueryOptions(job.projectId))
  const task = data?.project.task
  const descriptor = task ? getTaskDescriptor(task) : null
  const classTask = !!descriptor?.annotation.requiresLabelClasses && descriptor.annotation.type !== 'bounding_box'
  const [showFailures, setShowFailures] = useState(false)

  const ok = job.failed === 0 && job.status === 'done'
  const headline =
    job.status === 'cancelled'
      ? `Cancelled — ${job.uploaded.toLocaleString()} added`
      : job.failed > 0
        ? `${job.uploaded.toLocaleString()} added, ${job.failed.toLocaleString()} failed`
        : job.uploaded > 0
          ? `Added ${plural(job.uploaded, 'file')}`
          : 'Nothing new to add'

  const splitSegments = (['train', 'validation', 'test'] as const).map((split) => ({
    key: split,
    label: split,
    value: job.bySplit[split],
    color: SPLIT_COLORS[split],
  }))

  const dismiss = () => {
    if (job.retryable === 0) return dismissJob(job.id)
    modals.openConfirmModal({
      title: 'Dismiss this result?',
      children: (
        <Text size="sm">
          {plural(job.retryable, 'file')} {job.status === 'cancelled' ? 'were not sent' : 'failed'}. Dismissing forgets
          them, and they would have to be added again.
        </Text>
      ),
      labels: { confirm: 'Dismiss', cancel: 'Keep' },
      confirmProps: { color: 'red' },
      onConfirm: () => dismissJob(job.id),
    })
  }

  return (
    <Stack gap={8}>
      <Group justify="space-between" wrap="nowrap" align="flex-start">
        <Group gap={6} wrap="nowrap" style={{ minWidth: 0 }}>
          {ok ? (
            <CheckCircleIcon size={18} weight="fill" color="var(--mantine-color-green-6)" />
          ) : (
            <WarningCircleIcon size={18} weight="fill" color="var(--mantine-color-yellow-6)" />
          )}
          <Text size="sm" fw={600}>
            {headline}
          </Text>
        </Group>
        <Tooltip label="Dismiss">
          <ActionIcon size="sm" variant="subtle" color="gray" aria-label="Dismiss" onClick={dismiss}>
            <XIcon size={14} />
          </ActionIcon>
        </Tooltip>
      </Group>

      {job.uploaded > 0 && (
        <Stack gap={4}>
          <ProportionBar segments={splitSegments} height={8} />
          <Group gap="sm">
            {splitSegments.map((s) => (
              <Text key={s.key} size="xs" c="dimmed">
                {s.label} <b>{s.value.toLocaleString()}</b>
              </Text>
            ))}
          </Group>
        </Stack>
      )}

      <Group gap={6} wrap="wrap">
        {job.duplicates > 0 && (
          <Badge size="sm" variant="light" color="gray">
            {job.duplicates.toLocaleString()} already in the dataset
          </Badge>
        )}
        {job.createdClasses > 0 && (
          <Badge size="sm" variant="light" color="teal">
            {plural(job.createdClasses, 'class')} created
          </Badge>
        )}
      </Group>

      {job.failed > 0 && (
        <Stack gap={4}>
          <Button
            size="compact-xs"
            variant="subtle"
            color="red"
            onClick={() => setShowFailures((v) => !v)}
            style={{ alignSelf: 'flex-start' }}
          >
            {showFailures ? 'Hide' : 'Show'} what failed
          </Button>
          <Collapse expanded={showFailures}>
            <ScrollArea.Autosize mah={140}>
              <Stack gap={4}>
                {job.failures.slice(0, 50).map((f) => (
                  <div key={f.id}>
                    <Text size="xs" fw={500} style={{ wordBreak: 'break-all' }}>
                      {f.name}
                    </Text>
                    <Text size="xs" c="red.7">
                      {f.reason}
                    </Text>
                  </div>
                ))}
                {job.failures.length > 50 && (
                  <Text size="xs" c="dimmed">
                    …and {job.failures.length - 50} more
                  </Text>
                )}
              </Stack>
            </ScrollArea.Autosize>
          </Collapse>
        </Stack>
      )}

      {classTask && job.uploaded > 0 && <ClassBalance projectId={job.projectId} />}

      <Group gap={6} wrap="wrap">
        {job.retryable > 0 && (
          <Button size="compact-sm" variant="filled" onClick={() => retryJob(job.id)}>
            {job.status === 'cancelled' ? 'Resume' : 'Retry'} {job.retryable.toLocaleString()}
          </Button>
        )}
        {job.unlabelled > 0 && (
          <LinkButton
            size="compact-sm"
            variant="light"
            to="/project/$projectId/data"
            params={{ projectId: job.projectId }}
            search={
              classTask
                ? { page: 1, perPage: 20, classId: 'unassigned' as const }
                : { page: 1, perPage: 20, labeled: 'unlabeled' as const }
            }
          >
            Label {job.unlabelled.toLocaleString()} unlabelled
          </LinkButton>
        )}
        {job.uploaded > 0 && (
          <LinkButton
            size="compact-sm"
            variant="light"
            to="/project/$projectId/snapshots/new"
            params={{ projectId: job.projectId }}
          >
            Create snapshot
          </LinkButton>
        )}
      </Group>
    </Stack>
  )
}

export function UploadTray() {
  const jobs = useUploadJobs()

  // A reload ends every running job: files that were not sent yet are lost with the page.
  const running = jobs.some((job) => job.status === 'running')
  useEffect(() => {
    if (!running) return
    const warn = (event: BeforeUnloadEvent) => {
      if (hasRunningJob()) event.preventDefault()
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [running])

  if (jobs.length === 0) return null
  return (
    <Stack
      gap="xs"
      role="region"
      aria-label="Uploads"
      style={{
        position: 'fixed',
        right: 16,
        bottom: 16,
        width: 'min(360px, calc(100vw - 32px))',
        maxHeight: 'calc(100vh - 96px)',
        overflowY: 'auto',
        // Above the add-data drawer, so a batch started there can be watched without closing it.
        zIndex: 400,
      }}
    >
      {[...jobs].reverse().map((job) => (
        <Paper key={job.id} withBorder shadow="md" p="sm" radius="md">
          {job.status === 'running' ? <RunningJob job={job} /> : <FinishedJob job={job} />}
        </Paper>
      ))}
    </Stack>
  )
}

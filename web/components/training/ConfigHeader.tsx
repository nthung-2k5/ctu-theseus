import { Alert, Badge, Button, Group, Paper, Select, Text, TextInput, Tooltip } from '@mantine/core'
import { LockIcon, RocketLaunchIcon } from '@phosphor-icons/react'
import { MODALITY_META } from '@public/lib/modality'
import { getTaskDescriptor, isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'
import { Link } from '@tanstack/react-router'
import type { ReactNode } from 'react'

export interface BackendOption {
  id: string
  label: string
}

/**
 * The strip at the top of "new run" and "new sweep": name, the task (fixed by the project), the
 * snapshot to train on, the trainer backend, and the class count when the task uses classes.
 * `extras` adds mode-specific fields after them; `right` sits at the far end (the Simple/Advanced switch).
 */
export function ConfigHeader({
  project,
  nameLabel,
  namePlaceholder,
  name,
  onNameChange,
  versionId,
  onVersionChange,
  backends,
  backendId,
  onBackendChange,
  extras,
  right,
}: {
  project: ProjectDetail
  nameLabel: string
  namePlaceholder?: string
  name: string
  onNameChange: (name: string) => void
  versionId: string | null
  onVersionChange: (id: string | null) => void
  backends: BackendOption[]
  backendId: string | null
  onBackendChange: (id: string | null) => void
  extras?: ReactNode
  right?: ReactNode
}) {
  const descriptor = getTaskDescriptor(project.task)
  const modality = MODALITY_META[descriptor.modality]
  const versions = (project.dataset?.versions ?? []).filter((v) => v.status === 'ready')
  const selected = versions.find((v) => v.id === versionId)
  const classCount = project.dataset?.classes?.length ?? 0

  return (
    <Paper p="sm">
      <Group align="flex-end" gap="md">
        <TextInput
          size="xs"
          label={nameLabel}
          placeholder={namePlaceholder}
          w={240}
          value={name}
          onChange={(e) => onNameChange(e.currentTarget.value)}
          error={name.trim() ? undefined : 'required'}
        />
        <div>
          <Text size="xs" fw={500} mb={4}>
            Task
          </Text>
          <Tooltip label="Fixed by the project">
            <Badge size="lg" variant="outline" color="gray" leftSection={<LockIcon size={11} />}>
              {modality.label} · {descriptor.label}
            </Badge>
          </Tooltip>
        </div>
        <Select
          size="xs"
          label="Snapshot"
          w={210}
          allowDeselect={false}
          searchable
          placeholder={versions.length === 0 ? 'No ready snapshots' : 'Select a snapshot'}
          disabled={versions.length === 0}
          data={versions.map((v) => ({
            value: v.id,
            label: `${v.versionTag}${v.augmentedCount ? ` · +${v.augmentedCount.toLocaleString()} aug` : ''}`,
          }))}
          value={versionId}
          onChange={onVersionChange}
        />
        {backends.length > 1 && (
          <Select
            size="xs"
            label="Trainer backend"
            w={170}
            allowDeselect={false}
            data={backends.map((b) => ({ value: b.id, label: b.label }))}
            value={backendId}
            onChange={onBackendChange}
          />
        )}
        {extras}
        {isClassificationTask(project.task) && (
          <Tooltip label="Manage classes">
            <Link to="/project/$projectId/classes" params={{ projectId: project.id }}>
              <Badge size="lg" color={classCount < 2 ? 'yellow' : 'teal'} style={{ cursor: 'pointer' }}>
                {classCount} classes
              </Badge>
            </Link>
          </Tooltip>
        )}
        {right && <div style={{ marginLeft: 'auto' }}>{right}</div>}
      </Group>
      {selected && (
        <Text size="xs" c="dimmed" mt="xs" className="tnum">
          Training on {selected.versionTag}: {(selected.itemCount ?? 0).toLocaleString()} items
          {selected.augmentedCount ? ` (${selected.augmentedCount.toLocaleString()} augmented)` : ''} · #
          {selected.id.slice(0, 8)}
        </Text>
      )}
    </Paper>
  )
}

/** "Before you can launch" list plus the launch button, right-aligned under the form. */
export function LaunchBar({
  problems,
  label,
  icon,
  loading,
  onLaunch,
}: {
  problems: string[]
  label: string
  icon?: ReactNode
  loading?: boolean
  onLaunch: () => void
}) {
  return (
    <>
      {problems.length > 0 && (
        <Alert color="yellow" p="xs" title="Before you can launch">
          {problems.map((p) => (
            <Text key={p} size="xs">
              • {p}
            </Text>
          ))}
        </Alert>
      )}
      <Group justify="flex-end">
        <Button
          leftSection={icon ?? <RocketLaunchIcon size={15} />}
          disabled={problems.length > 0}
          loading={loading}
          onClick={onLaunch}
        >
          {label}
        </Button>
      </Group>
    </>
  )
}

import { Alert, Badge, Button, Group, Paper, Select, Text, TextInput, Tooltip } from '@mantine/core'
import { RocketLaunchIcon } from '@phosphor-icons/react'
import { LabelWithTip } from '@public/components/ui'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'
import { Link } from '@tanstack/react-router'
import type { ReactNode } from 'react'

export interface BackendOption {
  id: string
  label: string
}

/**
 * The strip at the top of "new run" and "new sweep": name, the snapshot to train on, the trainer backend, and the class count when the task uses classes.
 * `extras` adds mode-specific fields after them; `right` sits at the far end.
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
  hideSnapshot,
  hideClassCount,
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
  /** The snapshot is chosen elsewhere on the page (a list), so leave out its dropdown and summary line. */
  hideSnapshot?: boolean
  /** Leave out the class-count badge (the launch checks still require at least 2 classes). */
  hideClassCount?: boolean
  extras?: ReactNode
  right?: ReactNode
}) {
  const versions = (project.dataset?.versions ?? []).filter((v) => v.status === 'ready')
  const selected = versions.find((v) => v.id === versionId)
  const classCount = project.dataset?.classes?.length ?? 0

  return (
    <Paper p="sm">
      <Group align="flex-end" gap="md">
        <TextInput
          size="xs"
          label={
            <LabelWithTip
              label={nameLabel}
              tip="A name so you can recognise this later in your list of training runs. It does not affect how the model trains."
            />
          }
          placeholder={namePlaceholder}
          w={hideSnapshot ? undefined : 240}
          style={hideSnapshot ? { flex: 1 } : undefined}
          value={name}
          onChange={(e) => onNameChange(e.currentTarget.value)}
          error={name.trim() ? undefined : 'required'}
        />
        {!hideSnapshot && (
          <Select
            size="xs"
            label={
              <LabelWithTip
                label="Snapshot"
                tip="A frozen copy of your dataset that the model learns from. Because it never changes, results stay comparable between runs. Make a new snapshot after you change your data."
              />
            }
            w={210}
            allowDeselect={false}
            searchable
            placeholder={versions.length === 0 ? 'No ready snapshots' : 'Select a snapshot'}
            disabled={versions.length === 0}
            data={versions.map((v) => ({
              value: v.id,
              label: `${v.versionTag}${v.preprocessedCount ? ` · ~${v.preprocessedCount.toLocaleString()} pre` : ''}${v.augmentedCount ? ` · +${v.augmentedCount.toLocaleString()} aug` : ''}`,
            }))}
            value={versionId}
            onChange={onVersionChange}
          />
        )}
        {backends.length > 1 && (
          <Select
            size="xs"
            label={
              <LabelWithTip
                label="Trainer backend"
                tip="The software that does the training. Different ones can offer different models and settings."
              />
            }
            w={170}
            allowDeselect={false}
            data={backends.map((b) => ({ value: b.id, label: b.label }))}
            value={backendId}
            onChange={onBackendChange}
          />
        )}
        {extras}
        {!hideClassCount && isClassificationTask(project.task) && (
          <Tooltip
            label="The categories the model learns to tell apart. You need at least 2. Click to manage them."
            multiline
            w={240}
          >
            <Link to="/project/$projectId/classes" params={{ projectId: project.id }}>
              <Badge size="lg" color={classCount < 2 ? 'yellow' : 'teal'} style={{ cursor: 'pointer' }}>
                {classCount} classes
              </Badge>
            </Link>
          </Tooltip>
        )}
        {right && <div style={{ marginLeft: 'auto' }}>{right}</div>}
      </Group>
      {!hideSnapshot && selected && (
        <Text size="xs" c="dimmed" mt="xs" className="tnum">
          Training on {selected.versionTag}: {(selected.itemCount ?? 0).toLocaleString()} items
          {selected.preprocessedCount ? ` (${selected.preprocessedCount.toLocaleString()} preprocessed)` : ''}
          {selected.augmentedCount ? ` (${selected.augmentedCount.toLocaleString()} augmented)` : ''}
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

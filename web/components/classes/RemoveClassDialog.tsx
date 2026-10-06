import { Button, Group, Select, Stack, Text } from '@mantine/core'
import { modals } from '@mantine/modals'
import { useState } from 'react'

export interface MergeTarget {
  classId: string
  name: string
}

const NONE = '__none__'

function RemoveClassBody({
  name,
  count,
  targets,
  initialTarget,
  onConfirm,
  onCancel,
}: {
  name: string
  /** Items in the draft with this class; undefined while the counts are still loading. */
  count: number | undefined
  targets: MergeTarget[]
  initialTarget: string | null
  onConfirm: (target: string | null) => void
  onCancel: () => void
}) {
  const [target, setTarget] = useState<string>(initialTarget ?? NONE)
  const items = count === undefined ? 'Items' : `${count.toLocaleString()} item${count === 1 ? '' : 's'}`
  const merging = target !== NONE

  return (
    <Stack gap="sm">
      <Text size="sm">
        {items} in the draft {count === 1 ? 'is' : 'are'} labelled "{name}". Choose what happens to{' '}
        {count === 1 ? 'it' : 'them'}. Nothing changes until you press Save.
      </Text>
      <Select
        label="Items of this class"
        allowDeselect={false}
        searchable={targets.length > 8}
        value={target}
        onChange={(v) => setTarget(v ?? NONE)}
        data={[
          { value: NONE, label: 'Leave them unlabelled' },
          ...targets.map((t) => ({ value: t.classId, label: `Move to "${t.name}"` })),
        ]}
      />
      <Text size="xs" c="dimmed">
        Existing snapshots keep their own copy of the class. New snapshots will not include "{name}"
        {merging ? '' : ', and these items will be unlabelled in them'}.
      </Text>
      <Group justify="flex-end">
        <Button variant="default" onClick={onCancel}>
          Keep class
        </Button>
        <Button color={merging ? 'primary' : 'red'} onClick={() => onConfirm(merging ? target : null)}>
          {merging ? 'Merge and remove' : 'Remove class'}
        </Button>
      </Group>
    </Stack>
  )
}

/**
 * Ask what should happen to a saved class's items before it is removed: leave them unlabelled (the old
 * behaviour) or move them into another class. `onConfirm` gets the chosen target class id, or null for
 * "leave unlabelled".
 */
export function openRemoveClass({
  name,
  count,
  targets,
  initialTarget = null,
  onConfirm,
}: {
  name: string
  count: number | undefined
  targets: MergeTarget[]
  initialTarget?: string | null
  onConfirm: (target: string | null) => void
}) {
  const id = modals.open({
    title: `Remove "${name}"?`,
    children: (
      <RemoveClassBody
        name={name}
        count={count}
        targets={targets}
        initialTarget={initialTarget}
        onCancel={() => modals.close(id)}
        onConfirm={(target) => {
          modals.close(id)
          onConfirm(target)
        }}
      />
    ),
  })
}

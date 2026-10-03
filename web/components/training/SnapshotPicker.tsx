import { Badge, Group, ScrollArea, Stack, Text, UnstyledButton } from '@mantine/core'
import type { DatasetVersion } from '@public/store/types'

/** What a snapshot is, in words for someone who has never trained a model. */
export const SNAPSHOT_HELP =
  'A frozen copy of your dataset that the model learns from. Because it never changes, results stay comparable between runs. Make a new snapshot after you change your data.'

/**
 * The snapshot chooser: a scrolling list of the project's snapshots that finished building, newest first,
 * each with its item counts. Only these can be trained on, so the caller passes only ready versions.
 */
export function SnapshotPicker({
  versions,
  versionId,
  onChange,
  height = 300,
}: {
  versions: DatasetVersion[]
  versionId: string | null
  onChange: (id: string) => void
  height?: number
}) {
  const newestFirst = [...versions].reverse()

  return (
    <ScrollArea h={height} type="auto" offsetScrollbars>
      <Stack gap={6} role="radiogroup" aria-label="Dataset snapshot">
        {newestFirst.length === 0 && (
          <Text size="xs" c="dimmed">
            No ready snapshots. Build one first: there is nothing to train on yet.
          </Text>
        )}
        {newestFirst.map((v) => {
          const selected = v.id === versionId
          return (
            <UnstyledButton
              key={v.id}
              onClick={() => onChange(v.id)}
              role="radio"
              aria-checked={selected}
              p="xs"
              style={{
                borderRadius: 'var(--mantine-radius-sm)',
                border: `1px solid ${selected ? 'var(--mantine-color-primary-6)' : 'var(--mantine-color-default-border)'}`,
                background: selected ? 'var(--mantine-color-default-hover)' : 'transparent',
              }}
            >
              <Group gap={6} wrap="nowrap">
                <Text size="sm" fw={500} truncate style={{ minWidth: 0 }}>
                  {v.versionTag ?? v.id.slice(0, 8)}
                </Text>
                {v.preprocessedCount > 0 && (
                  <Badge size="xs" variant="light" color="cyan" style={{ flexShrink: 0 }}>
                    {v.preprocessedCount.toLocaleString()} preprocessed
                  </Badge>
                )}
                {v.augmentedCount > 0 && (
                  <Badge size="xs" variant="light" color="grape" style={{ flexShrink: 0 }}>
                    +{v.augmentedCount.toLocaleString()} augmented
                  </Badge>
                )}
              </Group>
              <Text size="xs" c="dimmed" className="tnum">
                {(v.itemCount ?? 0).toLocaleString()} items
                {v.classCount ? ` · ${v.classCount} classes` : ''}
                {v.builtAt ? ` · built ${new Date(v.builtAt).toLocaleDateString()}` : ''}
              </Text>
            </UnstyledButton>
          )
        })}
      </Stack>
    </ScrollArea>
  )
}

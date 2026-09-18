import { Progress, Select, Tooltip } from '@mantine/core'
import { FunnelIcon } from '@phosphor-icons/react'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { DatasetVersion } from '@public/store/types'

export const SPLIT_TYPES = ['train', 'validation', 'test'] as const

/**
 * Per-split membership counts, computed server-side (see
 * `server/routes/projects.ts`). This used to tally a raw `version.items`
 * array, which meant every project navigation shipped the entire membership
 * table just so the UI could count it.
 */
export const splitCounts = (version: DatasetVersion) =>
  version.splitCounts ??
  (Object.fromEntries(SPLIT_TYPES.map((s) => [s, 0])) as Record<(typeof SPLIT_TYPES)[number], number>)

/* ── Split progress bar (top of the item list — visualizes train/validation/test membership) ── */
export const SplitProgressBar = ({ version }: { version: DatasetVersion }) => {
  const counts = splitCounts(version)
  const total = SPLIT_TYPES.reduce((sum, s) => sum + counts[s], 0)

  return (
    <Progress.Root size={12} radius="xl">
      {total > 0 ? (
        SPLIT_TYPES.filter((s) => counts[s] > 0).map((splitType) => (
          <Tooltip
            label={`${splitType[0].toUpperCase()}${splitType.slice(1)}: ${counts[splitType]}`}
            withArrow
            key={splitType}
          >
            <Progress.Section value={(counts[splitType] / total) * 100} color={SPLIT_COLORS[splitType] ?? 'gray'} />
          </Tooltip>
        ))
      ) : (
        <Progress.Section value={100} color="gray.3" />
      )}
    </Progress.Root>
  )
}

/* ── Split Filter (select dropdown — filter items by split) ── */
export const SplitFilter = ({
  version,
  value,
  onChange,
}: {
  version: DatasetVersion
  value: (typeof SPLIT_TYPES)[number] | null
  onChange: (splitType: (typeof SPLIT_TYPES)[number] | null) => void
}) => {
  const counts = splitCounts(version)

  return (
    <Select
      placeholder="Filter by split"
      data={SPLIT_TYPES.map((s) => ({ value: s, label: `${s[0].toUpperCase()}${s.slice(1)} (${counts[s]})` }))}
      value={value}
      onChange={(v) => onChange((v as (typeof SPLIT_TYPES)[number] | null) ?? null)}
      clearable
      w={200}
      leftSection={<FunnelIcon size={14} />}
    />
  )
}

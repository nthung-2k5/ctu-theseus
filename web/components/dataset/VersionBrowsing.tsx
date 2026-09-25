import { Select } from '@mantine/core'
import { FunnelIcon } from '@phosphor-icons/react'
import { ProportionBar } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import type { DatasetVersion } from '@public/store/types'

export const SPLIT_TYPES = ['train', 'validation', 'test'] as const

/**
 * Per-split membership counts, computed server-side (see
 * `ai_service/theseus/routers/projects.py`). This used to tally a raw `version.items`
 * array, which meant every project navigation shipped the entire membership
 * table just so the UI could count it.
 */
export const splitCounts = (version: DatasetVersion) =>
  Object.fromEntries(SPLIT_TYPES.map((s) => [s, version.splitCounts?.[s] ?? 0])) as Record<
    (typeof SPLIT_TYPES)[number],
    number
  >

/* ── Split bar (top of the item list — visualizes train/validation/test membership) ── */
export const SplitProgressBar = ({ version }: { version: DatasetVersion }) => {
  const counts = splitCounts(version)
  return (
    <ProportionBar
      segments={SPLIT_TYPES.map((s) => ({
        key: s,
        label: `${s[0].toUpperCase()}${s.slice(1)}`,
        value: counts[s],
        color: SPLIT_COLORS[s] ?? 'gray',
      }))}
    />
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

import { Select } from '@mantine/core'
import { TagIcon } from '@phosphor-icons/react'
import type { LabelClass } from '@public/store/types'

/**
 * Filters an item list down to one label class. Renders nothing when there
 * are no classes to filter by — callers don't need to gate on `classes.length`
 * themselves.
 *
 * `counts` (classId -> item count, plus the `unassigned` sentinel) is
 * optional — when supplied, each option is labeled "Name (10)"; omit it (or
 * pass no matching entry) and the option just shows the plain name.
 */
export function ClassFilter({
  classes,
  value,
  onChange,
  counts,
}: {
  classes: LabelClass[]
  value: string | null
  onChange: (classId: string | null) => void
  counts?: Record<string, number>
}) {
  if (classes.length === 0) return null

  const withCount = (label: string, id: string) => (counts && id in counts ? `${label} (${counts[id]})` : label)

  return (
    <Select
      placeholder="Filter by class"
      data={[
        { value: 'unassigned', label: withCount('Unassigned', 'unassigned') },
        ...classes.map((c) => ({ value: c.classId, label: withCount(c.name, c.classId) })),
      ]}
      value={value}
      onChange={onChange}
      clearable
      w={220}
      leftSection={<TagIcon size={14} />}
    />
  )
}

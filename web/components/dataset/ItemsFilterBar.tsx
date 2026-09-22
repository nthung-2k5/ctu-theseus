import { Group, SegmentedControl, Select, TextInput } from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { MagnifyingGlassIcon, SortAscendingIcon } from '@phosphor-icons/react'
import { ClassFilter } from '@public/components/dataset/ClassFilter'
import { type SPLIT_TYPES, SplitFilter } from '@public/components/dataset/VersionBrowsing'
import type { DatasetVersion, LabelClass } from '@public/store/types'
import { useEffect, useState } from 'react'

export type ItemSort = 'newest' | 'oldest' | 'filename'
export type ItemOrigin = 'original' | 'augmented'

const ORIGIN_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'original', label: 'Original' },
  { value: 'augmented', label: 'Augmented' },
]

const SORT_OPTIONS: { value: ItemSort; label: string }[] = [
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'filename', label: 'Filename' },
]

/**
 * Filter/sort row for an item list: search, split, class, and sort — all in
 * one row below the split progress bar. Shared by the Dataset draft view and
 * the Snapshot detail view. The origin filter (original vs augmented items)
 * only appears for a snapshot that was built with augmentation.
 */
export function ItemsFilterBar({
  version,
  split,
  onSplitChange,
  classes,
  classId,
  onClassChange,
  classCounts,
  search,
  onSearchChange,
  sort,
  onSortChange,
  origin,
  onOriginChange,
}: {
  version: DatasetVersion
  split: (typeof SPLIT_TYPES)[number] | null
  onSplitChange: (split: (typeof SPLIT_TYPES)[number] | null) => void
  classes: LabelClass[]
  classId: string | null
  onClassChange: (classId: string | null) => void
  classCounts?: Record<string, number>
  search: string
  onSearchChange: (search: string) => void
  sort: ItemSort
  onSortChange: (sort: ItemSort) => void
  origin?: ItemOrigin | null
  onOriginChange?: (origin: ItemOrigin | null) => void
}) {
  const [searchDraft, setSearchDraft] = useState(search)
  const [debounced] = useDebouncedValue(searchDraft, 300)

  // Resync when `search` changes from outside this component (e.g. browser
  // back/forward navigation), without fighting the debounce below.
  useEffect(() => setSearchDraft(search), [search])

  // biome-ignore lint/correctness/useExhaustiveDependencies: fire only when the debounced value settles, not on every render
  useEffect(() => {
    if (debounced !== search) onSearchChange(debounced)
  }, [debounced])

  return (
    <Group gap="sm" wrap="wrap">
      <TextInput
        placeholder="Search filename"
        leftSection={<MagnifyingGlassIcon size={14} />}
        value={searchDraft}
        onChange={(e) => setSearchDraft(e.currentTarget.value)}
        style={{ flex: 1, minWidth: 180 }}
      />
      {onOriginChange && version.augmentedCount > 0 && (
        <SegmentedControl
          data={ORIGIN_OPTIONS}
          value={origin ?? 'all'}
          onChange={(v) => onOriginChange(v === 'all' ? null : (v as ItemOrigin))}
        />
      )}
      <SplitFilter version={version} value={split} onChange={onSplitChange} />
      <ClassFilter classes={classes} value={classId} onChange={onClassChange} counts={classCounts} />
      <Select
        w={160}
        leftSection={<SortAscendingIcon size={14} />}
        data={SORT_OPTIONS}
        value={sort}
        onChange={(v) => v && onSortChange(v as ItemSort)}
        allowDeselect={false}
      />
    </Group>
  )
}

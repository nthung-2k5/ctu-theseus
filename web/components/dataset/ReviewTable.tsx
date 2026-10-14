/**
 * The review table: everything staged, as one flat, filterable, virtualized list, whatever it is made of.
 *
 * Files and inline rows are both shown as `ReviewRow`s (see `lib/upload/review.ts`), so the same table serves every
 * task. Each row's split and class are editable in place; selecting rows (or ticking the header to take every row the
 * current filter shows) opens bulk actions. It edits nothing itself: changes go out through `onEdit` / `onRemove`.
 */

import { ActionIcon, Badge, Button, Checkbox, Group, NativeSelect, Stack, Text, Tooltip } from '@mantine/core'
import { TrashIcon, WarningIcon, XIcon } from '@phosphor-icons/react'
import { STAGED_SPLIT_OPTIONS } from '@public/lib/constants'
import { formatBytes } from '@public/lib/format'
import {
  countRows,
  FILTER_LABELS,
  filterRows,
  type ReviewFilter,
  type ReviewRow,
  rowNote,
} from '@public/lib/upload/review'
import type { StagedSplit } from '@public/lib/upload/types'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type ReactNode, useMemo, useRef, useState } from 'react'
import { Thumb } from './Thumb'

const ROW_HEIGHT = 48
const THUMB = 36

export interface RowEdit {
  split?: StagedSplit
  classKey?: string | null
}

interface ClassOption {
  value: string
  label: string
}

const FILTERS = Object.keys(FILTER_LABELS) as ReviewFilter[]

function gridColumns(showClass: boolean, showSize: boolean, showThumb: boolean) {
  return [
    '28px',
    showThumb ? `${THUMB}px` : null,
    'minmax(0,1fr)',
    showClass ? '170px' : null,
    '128px',
    showSize ? '72px' : null,
    '60px',
  ]
    .filter(Boolean)
    .join(' ')
}

const NOTE_COLOR = { error: 'red.7', warning: 'yellow.8', muted: 'dimmed' } as const

function TableRow({
  row,
  columns,
  classOptions,
  showClass,
  showSize,
  showThumb,
  selected,
  disabled,
  onToggle,
  onEdit,
  onRemove,
}: {
  row: ReviewRow
  columns: string
  classOptions: ClassOption[]
  showClass: boolean
  showSize: boolean
  showThumb: boolean
  selected: boolean
  disabled: boolean
  onToggle: (id: string) => void
  onEdit: (ids: string[], edit: RowEdit) => void
  onRemove: (ids: string[]) => void
}) {
  const errored = row.issues.some((i) => i.severity === 'error')
  const note = rowNote(row)
  // The note already says the worst thing; the icon is only there when there is more to read.
  const messages = [...row.issues.map((i) => i.message), ...(row.skip ? [`${row.skip}, so it is not sent`] : [])]
  // A class key nobody offers (its folder was renamed away) still has to show as what it is.
  const known = row.classKey === null || classOptions.some((o) => o.value === row.classKey)
  return (
    <div
      className="items-center gap-2 px-2"
      style={{
        display: 'grid',
        gridTemplateColumns: columns,
        height: ROW_HEIGHT,
        background: selected ? 'var(--mantine-primary-color-light)' : undefined,
        borderBottom: '1px solid var(--mantine-color-default-border)',
      }}
    >
      <Checkbox
        size="xs"
        aria-label={`Select ${row.name}`}
        checked={selected}
        onChange={() => onToggle(row.id)}
        disabled={disabled}
      />
      {showThumb && <Thumb file={row.thumb} media={row.media} size={THUMB} />}
      <div style={{ minWidth: 0 }}>
        <Text
          size="sm"
          truncate
          td={errored ? 'line-through' : undefined}
          c={errored || row.skip ? 'dimmed' : undefined}
        >
          {row.name}
        </Text>
        {note ? (
          <Text size="xs" truncate c={NOTE_COLOR[note.tone]} title={note.text}>
            {note.text}
          </Text>
        ) : (
          row.detail && (
            <Text size="xs" c="dimmed" truncate>
              {row.detail}
            </Text>
          )
        )}
      </div>
      {/* A row that can never upload has nowhere to go, so it gets no class or split to edit. */}
      {showClass &&
        (errored ? (
          <span />
        ) : (
          <NativeSelect
            size="xs"
            aria-label={`Class of ${row.name}`}
            data={[
              { value: '', label: 'No class' },
              ...classOptions,
              ...(known ? [] : [{ value: row.classKey as string, label: 'Unknown class' }]),
            ]}
            value={row.classKey ?? ''}
            onChange={(event) => onEdit([row.id], { classKey: event.currentTarget.value || null })}
            disabled={disabled}
          />
        ))}
      {errored ? (
        <span />
      ) : (
        <NativeSelect
          size="xs"
          aria-label={`Split of ${row.name}`}
          data={STAGED_SPLIT_OPTIONS}
          value={row.split}
          onChange={(event) => onEdit([row.id], { split: event.currentTarget.value as StagedSplit })}
          disabled={disabled}
        />
      )}
      {showSize && (
        <Text size="xs" c="dimmed" ta="right">
          {row.size === undefined ? '' : formatBytes(row.size)}
        </Text>
      )}
      <Group gap={2} wrap="nowrap" justify="flex-end">
        {messages.length > 1 && (
          <Tooltip multiline maw={320} label={messages.map((m) => <div key={m}>{m}</div>)}>
            <WarningIcon
              size={16}
              weight="fill"
              color={errored ? 'var(--mantine-color-red-6)' : 'var(--mantine-color-yellow-6)'}
              aria-label="More problems"
            />
          </Tooltip>
        )}
        <ActionIcon
          size="sm"
          variant="subtle"
          color="gray"
          aria-label={`Remove ${row.name}`}
          onClick={() => onRemove([row.id])}
          disabled={disabled}
        >
          <XIcon size={12} />
        </ActionIcon>
      </Group>
    </div>
  )
}

export function ReviewTable({
  rows,
  classOptions,
  showClass,
  onEdit,
  onRemove,
  disabled = false,
  emptyMessage,
}: {
  rows: ReviewRow[]
  /** Classes a row can be given: existing ones, plus folders about to become classes. */
  classOptions: ClassOption[]
  showClass: boolean
  onEdit: (ids: string[], edit: RowEdit) => void
  onRemove: (ids: string[]) => void
  disabled?: boolean
  emptyMessage: ReactNode
}) {
  // TanStack Virtual returns functions the React Compiler would otherwise memoize away, which also switches off its
  // automatic memoization here; the work that scales with the row count is memoized by hand below.
  'use no memo'

  const [filter, setFilter] = useState<ReviewFilter>('all')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const parentRef = useRef<HTMLDivElement>(null)

  const counts = useMemo(() => countRows(rows), [rows])
  // A filter that nothing matches any more (the last error was fixed) falls back to showing everything.
  const active: ReviewFilter = filter !== 'all' && counts[filter] === 0 ? 'all' : filter
  const shown = useMemo(() => filterRows(rows, active), [rows, active])
  const showSize = useMemo(() => rows.some((r) => r.size !== undefined), [rows])
  const showThumb = useMemo(() => rows.some((r) => r.media !== undefined), [rows])

  const virtualizer = useVirtualizer({
    count: shown.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
    getItemKey: (i) => shown[i].id,
  })

  // Rows that were uploaded or removed since they were ticked drop out of the selection.
  const selectedIds = useMemo(() => {
    if (selected.size === 0) return []
    const live = new Set(rows.map((r) => r.id))
    return [...selected].filter((id) => live.has(id))
  }, [selected, rows])
  const shownSelected = useMemo(() => shown.reduce((n, r) => n + (selected.has(r.id) ? 1 : 0), 0), [shown, selected])

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (!next.delete(id)) next.add(id)
      return next
    })
  const toggleAllShown = () =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (shownSelected === shown.length) for (const r of shown) next.delete(r.id)
      else for (const r of shown) next.add(r.id)
      return next
    })

  const columns = gridColumns(showClass, showSize, showThumb)

  if (rows.length === 0) {
    return (
      <Stack align="center" justify="center" gap="xs" h="100%" p="xl">
        {emptyMessage}
      </Stack>
    )
  }

  return (
    <Stack gap="xs" style={{ flex: 1, minHeight: 0 }}>
      <Group gap={6} wrap="wrap">
        {FILTERS.filter((f) => f === 'all' || counts[f] > 0).map((f) => (
          <Button
            key={f}
            size="compact-xs"
            variant={active === f ? 'filled' : 'light'}
            color={f === 'errors' ? 'red' : f === 'warnings' || f === 'unclassified' ? 'yellow' : undefined}
            onClick={() => setFilter(f)}
          >
            {FILTER_LABELS[f]} · {counts[f].toLocaleString()}
          </Button>
        ))}
      </Group>

      {selectedIds.length > 0 && (
        <Group gap="xs" wrap="wrap">
          <Badge variant="light">{selectedIds.length.toLocaleString()} selected</Badge>
          {!disabled && (
            <>
              <NativeSelect
                size="xs"
                aria-label="Set split of the selected"
                data={[{ value: '', label: 'Set split…' }, ...STAGED_SPLIT_OPTIONS]}
                value=""
                onChange={(event) => {
                  if (event.currentTarget.value)
                    onEdit(selectedIds, { split: event.currentTarget.value as StagedSplit })
                }}
              />
              {showClass && (
                <NativeSelect
                  size="xs"
                  aria-label="Set class of the selected"
                  data={[{ value: '', label: 'Set class…' }, { value: '__none__', label: 'No class' }, ...classOptions]}
                  value=""
                  onChange={(event) => {
                    const value = event.currentTarget.value
                    if (value) onEdit(selectedIds, { classKey: value === '__none__' ? null : value })
                  }}
                />
              )}
              <Button
                size="xs"
                variant="light"
                color="red"
                leftSection={<TrashIcon size={14} />}
                onClick={() => {
                  onRemove(selectedIds)
                  setSelected(new Set())
                }}
              >
                Remove
              </Button>
            </>
          )}
          <Button size="xs" variant="subtle" color="gray" onClick={() => setSelected(new Set())}>
            Clear selection
          </Button>
        </Group>
      )}

      <div
        className="items-center gap-2 px-2"
        style={{ display: 'grid', gridTemplateColumns: columns, paddingBlock: 4 }}
      >
        <Checkbox
          size="xs"
          aria-label="Select every row shown"
          checked={shown.length > 0 && shownSelected === shown.length}
          indeterminate={shownSelected > 0 && shownSelected < shown.length}
          onChange={toggleAllShown}
          disabled={disabled || shown.length === 0}
        />
        {showThumb && <span />}
        <Text size="xs" c="dimmed" fw={600}>
          Name
        </Text>
        {showClass && (
          <Text size="xs" c="dimmed" fw={600}>
            Class
          </Text>
        )}
        <Text size="xs" c="dimmed" fw={600}>
          Split
        </Text>
        {showSize && (
          <Text size="xs" c="dimmed" fw={600} ta="right">
            Size
          </Text>
        )}
        <span />
      </div>

      <section
        ref={parentRef}
        aria-label="Staged rows"
        style={{
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
          border: '1px solid var(--mantine-color-default-border)',
          borderRadius: 'var(--mantine-radius-md)',
        }}
      >
        {shown.length === 0 ? (
          <Text size="sm" c="dimmed" ta="center" p="lg">
            Nothing matches this filter.
          </Text>
        ) : (
          <ul
            style={{
              height: virtualizer.getTotalSize(),
              position: 'relative',
              margin: 0,
              padding: 0,
              listStyle: 'none',
            }}
          >
            {virtualizer.getVirtualItems().map((item) => (
              <li
                key={item.key}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: item.size,
                  transform: `translateY(${item.start}px)`,
                }}
              >
                <TableRow
                  row={shown[item.index]}
                  columns={columns}
                  classOptions={classOptions}
                  showClass={showClass}
                  showSize={showSize}
                  showThumb={showThumb}
                  selected={selected.has(shown[item.index].id)}
                  disabled={disabled}
                  onToggle={toggle}
                  onEdit={onEdit}
                  onRemove={onRemove}
                />
              </li>
            ))}
          </ul>
        )}
      </section>
    </Stack>
  )
}

import { Alert, Badge, Button, Group, Modal, Paper, Select, Stack, Text, TextInput } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { MagnifyingGlassIcon, PlusIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { EmptyState } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import { saveClasses } from '@public/lib/api/generated/classes/classes'
import type { ClassStatsResponse } from '@public/lib/api/generated/models'
import {
  attention,
  type DraftClass,
  filterRows,
  fromSaved,
  isDirty,
  looksLikeList,
  MAX_CLASSES,
  nameError,
  newKey,
  nextColor,
  norm,
  removedFrom,
  resolveMerges,
  restoreRow,
  type SavedClass,
  type SortMode,
  sortRows,
  summarize,
  toSaveBody,
  usedSplits,
} from '@public/lib/classDraft'
import { useMutation } from '@tanstack/react-query'
import { getRouteApi, useBlocker } from '@tanstack/react-router'
import { useMemo, useState } from 'react'
import { BulkAddForm } from './BulkAddForm'
import { ClassRowMenu } from './ClassRowMenu'
import { ClassStatsStrip } from './ClassStatsStrip'
import { ClassTable } from './ClassTable'
import { PendingRemovals } from './PendingRemovals'
import { openRemoveClass } from './RemoveClassDialog'

const routeApi = getRouteApi('/_app/project/$projectId/classes')

const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: 'created', label: 'Creation order' },
  { value: 'name', label: 'Name' },
  { value: 'count', label: 'Most items' },
  { value: 'attention', label: 'Needs attention first' },
]

/**
 * The class editor. Everything here edits a local draft; Save sends it in one batch, so a half-finished rename
 * or merge never reaches the server. Removing a class that has items can also say where those items go
 * (`mergeInto`), which the server applies in the same transaction as the save.
 */
export function ClassManager({
  projectId,
  saved,
  supportsMerge,
  showColor,
  stats: loadedStats,
  statsFailed,
  onRetryStats,
  onSaved,
}: {
  projectId: string
  saved: SavedClass[]
  /**
   * Whether a task's items carry one `classification` annotation each, which is what a merge relabels. Token
   * classification labels spans instead, so there a merge would delete the class and change nothing.
   */
  supportsMerge: boolean
  /** Whether classes are drawn on the item (boxes, masks), which is all a colour is for; hidden otherwise. */
  showColor: boolean
  /** Per-class counts for the draft; undefined until loaded. */
  stats: ClassStatsResponse | undefined
  statsFailed: boolean
  onRetryStats: () => void
  onSaved: () => void
}) {
  const navigate = routeApi.useNavigate()
  const baseline = useMemo(() => saved.map(fromSaved), [saved])
  const [rows, setRows] = useState<DraftClass[]>(baseline)
  /** removed class id -> the class id its items move to. Only meaningful for classes in `removed`. */
  const [mergeInto, setMergeInto] = useState<Record<string, string>>({})
  const [newName, setNewName] = useState('')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortMode>('created')
  const [attentionOnly, setAttentionOnly] = useState(false)
  const [bulk, setBulk] = useState<{ open: boolean; text: string }>({ open: false, text: '' })

  /* ── What the counts say ── */
  // A task that can't be counted per class (`counted: false`) gets no counts at all: zeros would read as
  // "every class is empty and every item unlabelled".
  const stats = loadedStats?.counted ? loadedStats : undefined
  const statsById = useMemo(() => (stats ? new Map(stats.stats.map((s) => [s.classId, s])) : undefined), [stats])
  const min = stats?.minItemsPerClass ?? 3
  const splitsInUse = useMemo(() => usedSplits(stats?.stats ?? []), [stats])
  const max = Math.max(1, ...(stats?.stats.map((s) => s.total) ?? []))
  const countOf = (row: DraftClass) => (row.classId ? statsById?.get(row.classId)?.total : undefined)
  const needsAttention = (row: DraftClass) =>
    !!row.classId && attention(statsById?.get(row.classId), min, splitsInUse) !== null
  const attentionCount = rows.filter(needsAttention).length

  /* ── What a save would do ── */
  const dirty = isDirty(baseline, rows)
  const removed = removedFrom(baseline, rows)
  const keptIds = new Set(rows.flatMap((r) => (r.classId ? [r.classId] : [])))
  const { merges, unresolved } = resolveMerges(
    removed.flatMap((r) => (r.classId ? [r.classId] : [])),
    mergeInto,
    keptIds,
  )
  const mergedItems = merges.reduce((n, m) => n + (statsById?.get(m.fromClassId)?.total ?? 0), 0)
  const diff = summarize(baseline, rows, merges.length)
  const hasErrors = rows.some((r) => nameError(r.name, rows, r.key))

  const save = useMutation({
    mutationFn: () => saveClasses(projectId, toSaveBody(rows, merges)),
    onSuccess: () => {
      onSaved()
      notifications.show({
        title: 'Classes saved',
        message:
          mergedItems > 0
            ? `${mergedItems.toLocaleString()} items moved to their new class. New snapshots will use this list.`
            : 'New snapshots will use this list. Existing snapshots are unchanged.',
        color: 'teal',
      })
    },
    onError: (error) =>
      notifications.show({
        title: 'Could not save classes',
        message: apiErrorMessage(error, 'Failed to save classes'),
        color: 'red',
      }),
  })

  // Leaving with unsaved edits asks first (in-app navigation and tab close).
  const blocker = useBlocker({ shouldBlockFn: () => dirty, enableBeforeUnload: () => dirty, withResolver: true })

  /* ── Editing ── */
  const patch = (key: string, change: Partial<DraftClass>) =>
    setRows((d) => d.map((c) => (c.key === key ? { ...c, ...change } : c)))

  const addNames = (names: string[]) =>
    setRows((d) => {
      let out = d
      for (const name of names) {
        if (out.length >= MAX_CLASSES) break
        out = [...out, { key: newKey(), name, description: '', color: nextColor(out) }]
      }
      return out
    })

  const addError = newName.trim() ? nameError(newName, rows) : null
  const submitNew = () => {
    const n = norm(newName)
    if (!n || addError) return
    addNames([n])
    setNewName('')
  }

  const removeRow = (row: DraftClass, target: string | null) => {
    setRows((d) => d.filter((c) => c.key !== row.key))
    const id = row.classId
    if (!id) return
    setMergeInto((m) => {
      const { [id]: _dropped, ...rest } = m
      return target ? { ...rest, [id]: target } : rest
    })
  }

  const targetsFor = (row: DraftClass) =>
    rows.flatMap((r) => (r.classId && r.key !== row.key ? [{ classId: r.classId, name: r.name }] : []))

  const requestRemove = (row: DraftClass) => {
    const n = countOf(row)
    // Nothing to decide about: never saved, known to hold no items, or no way to move them (see `supportsMerge`).
    if (!row.classId || n === 0 || !supportsMerge) return removeRow(row, null)
    openRemoveClass({
      name: row.name,
      count: n,
      targets: targetsFor(row),
      onConfirm: (target) => removeRow(row, target),
    })
  }

  const restore = (row: DraftClass) => {
    setRows((d) => restoreRow(baseline, d, row))
    const id = row.classId
    if (id)
      setMergeInto((m) => {
        const { [id]: _dropped, ...rest } = m
        return rest
      })
  }

  const destinationOf = (row: DraftClass) => {
    const merge = merges.find((m) => m.fromClassId === row.classId)
    return merge ? rows.find((r) => r.classId === merge.intoClassId)?.name : undefined
  }

  const discard = () => {
    setRows(baseline)
    setMergeInto({})
  }

  /* ── What is shown ── */
  const visible = sortRows(
    filterRows(attentionOnly ? rows.filter(needsAttention) : rows, query),
    sort,
    countOf,
    needsAttention,
  )
  const filtering = query.trim() !== '' || attentionOnly
  const isBlank = rows.length === 0 && removed.length === 0

  return (
    <div className="flex flex-col gap-3">
      {stats && (
        <ClassStatsStrip
          classCount={rows.length}
          total={stats.total}
          unassigned={stats.unassigned}
          retired={stats.retired}
          attentionCount={attentionCount}
          attentionOnly={attentionOnly}
          onToggleAttention={() => setAttentionOnly((v) => !v)}
        />
      )}
      {loadedStats && !loadedStats.counted && (
        <Text size="xs" c="dimmed">
          Per-class item counts aren't available for this task: its labels mark spans inside each item rather than
          giving each item one class.
        </Text>
      )}
      {statsFailed && (
        <Alert color="yellow" icon={<WarningCircleIcon size={16} />} p="xs" title="Item counts unavailable">
          <Group gap="xs">
            <Text size="xs">You can still edit classes; the per-class counts just didn't load.</Text>
            <Button size="compact-xs" variant="light" onClick={onRetryStats}>
              Try again
            </Button>
          </Group>
        </Alert>
      )}

      {hasErrors && (
        <Alert color="yellow" icon={<WarningCircleIcon size={16} />} p="xs">
          Fix the highlighted names before saving.
        </Alert>
      )}

      <Paper p="sm">
        <Group gap="xs" align="flex-start" wrap="wrap">
          <Group gap="xs" align="flex-start" wrap="nowrap" style={{ flex: '1 1 320px' }}>
            <TextInput
              style={{ flex: 1 }}
              size="xs"
              placeholder="New class name, then Enter"
              aria-label="New class name"
              value={newName}
              error={addError}
              onChange={(e) => setNewName(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  submitNew()
                }
              }}
              // Pasting a list (a column from a spreadsheet, a comma-separated line) is the bulk dialog's job.
              onPaste={(e) => {
                const text = e.clipboardData.getData('text')
                if (!looksLikeList(text)) return
                e.preventDefault()
                setBulk({ open: true, text })
              }}
            />
            <Button
              size="xs"
              leftSection={<PlusIcon size={14} />}
              onClick={submitNew}
              disabled={!newName.trim() || !!addError}
            >
              Add
            </Button>
            {!isBlank && (
              <Button size="xs" variant="default" onClick={() => setBulk({ open: true, text: '' })}>
                Add several…
              </Button>
            )}
          </Group>
          {rows.length > 1 && (
            <Group gap="xs" wrap="nowrap" style={{ flex: '1 1 300px' }}>
              <TextInput
                style={{ flex: 1 }}
                size="xs"
                placeholder="Search classes"
                aria-label="Search classes"
                leftSection={<MagnifyingGlassIcon size={14} />}
                value={query}
                onChange={(e) => setQuery(e.currentTarget.value)}
              />
              <Select
                size="xs"
                w={190}
                aria-label="Sort classes"
                data={SORT_OPTIONS}
                value={sort}
                allowDeselect={false}
                onChange={(v) => v && setSort(v as SortMode)}
              />
            </Group>
          )}
        </Group>
        {filtering && (
          <Text size="xs" c="dimmed" mt={6}>
            Showing {visible.length} of {rows.length}
            {attentionOnly ? ' (classes that need attention)' : ''}.{' '}
            <Button
              size="compact-xs"
              variant="subtle"
              onClick={() => {
                setQuery('')
                setAttentionOnly(false)
              }}
            >
              Clear filter
            </Button>
          </Text>
        )}
      </Paper>

      {isBlank ? (
        <Paper p="md">
          <Stack gap="sm">
            <EmptyState
              compact
              title="No classes yet"
              description="Add the labels your model should predict: type one above, or paste a list here."
            />
            <BulkAddForm draft={rows} onAdd={addNames} />
          </Stack>
        </Paper>
      ) : (
        <ClassTable
          rows={visible}
          allRows={rows}
          stats={statsById}
          max={max}
          min={min}
          splitsInUse={splitsInUse}
          showColor={showColor}
          onPatch={patch}
          emptyMessage={filtering ? 'No classes match this filter.' : 'No classes.'}
          renderMenu={(row) => (
            <ClassRowMenu
              name={row.name}
              canViewItems={!!row.classId}
              mergeTargets={supportsMerge && row.classId && (countOf(row) ?? 1) > 0 ? targetsFor(row) : []}
              onViewItems={() =>
                void navigate({
                  to: '/project/$projectId/dataset',
                  params: { projectId },
                  search: { page: 1, perPage: 20, classId: row.classId },
                })
              }
              onMerge={(target) => removeRow(row, target)}
              onRemove={() => requestRemove(row)}
            />
          )}
        />
      )}

      <PendingRemovals removed={removed} countOf={countOf} destinationOf={destinationOf} onRestore={restore} />

      {unresolved.length > 0 && (
        <Alert color="yellow" icon={<WarningCircleIcon size={16} />} p="xs">
          {unresolved.length} removed class{unresolved.length === 1 ? ' was' : 'es were'} set to merge into a class you
          have since removed too. Their items will be left unlabelled unless you restore that class.
        </Alert>
      )}

      <Paper p="sm" style={{ position: 'sticky', bottom: 8, zIndex: 3 }}>
        <Group justify="space-between">
          <Group gap="xs">
            {dirty ? <Badge color="yellow">unsaved changes</Badge> : <Badge color="teal">saved</Badge>}
            {diff.added > 0 && (
              <Text size="xs" c="teal">
                +{diff.added} added
              </Text>
            )}
            {diff.removed > 0 && (
              <Text size="xs" c="red">
                −{diff.removed} removed
              </Text>
            )}
            {diff.renamed > 0 && (
              <Text size="xs" c="yellow">
                {diff.renamed} renamed
              </Text>
            )}
            {diff.merged > 0 && (
              <Text size="xs" c="blue">
                {diff.merged} merged
              </Text>
            )}
          </Group>
          <Group gap="xs">
            <Button variant="default" size="compact-md" disabled={!dirty} onClick={discard}>
              Discard
            </Button>
            <Button
              size="compact-md"
              disabled={!dirty || hasErrors}
              loading={save.isPending}
              onClick={() => save.mutate()}
            >
              Save classes
            </Button>
          </Group>
        </Group>
      </Paper>
      <Text size="xs" c="dimmed">
        Class edits apply to <b>new</b> snapshots and runs. Existing snapshots are immutable and keep the classes they
        were built with.
      </Text>

      <Modal opened={bulk.open} onClose={() => setBulk({ open: false, text: '' })} title="Add several classes" centered>
        {/* key: reopening starts from the pasted text, not from whatever was typed last time */}
        <BulkAddForm
          key={bulk.text}
          draft={rows}
          initialText={bulk.text}
          onAdd={(names) => {
            addNames(names)
            setBulk({ open: false, text: '' })
          }}
          onCancel={() => setBulk({ open: false, text: '' })}
        />
      </Modal>

      <Modal
        opened={blocker.status === 'blocked'}
        onClose={() => blocker.reset?.()}
        title="Discard unsaved class changes?"
        centered
      >
        <Group justify="flex-end">
          <Button variant="default" onClick={() => blocker.reset?.()}>
            Keep editing
          </Button>
          <Button color="red" onClick={() => blocker.proceed?.()}>
            Discard and leave
          </Button>
        </Group>
      </Modal>
    </div>
  )
}

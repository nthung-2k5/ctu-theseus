/**
 * Classes page – the label-class registry for classification tasks.
 *
 * Edits (add, rename, recolour, describe, delete) build up in a local draft and are saved in one batch
 * (PUT /projects/:id/classes), so a half-finished rename never reaches the server. Classes are listed in
 * creation order. Snapshots are immutable: changes only apply to new snapshots and runs.
 */

import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  ColorInput,
  Group,
  Modal,
  Paper,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { InfoIcon, PlusIcon, TrashIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { MeterBar, PageHeader, QueryBoundary } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import { saveClasses } from '@public/lib/api/generated/classes/classes'
import {
  invalidateProjectScope,
  projectDetailQueryOptions,
  useDatasetHealth,
  useLabelClasses,
} from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import type { LabelClass } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, useBlocker } from '@tanstack/react-router'
import { useMemo, useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/classes')

/** Categorical palette, reused for new classes (first colour not already taken). */
const CLASS_PALETTE = [
  '#00afef',
  '#f59e0b',
  '#34d399',
  '#f472b6',
  '#a78bfa',
  '#4a89d1',
  '#fb7185',
  '#22d3ee',
  '#facc15',
  '#86efac',
  '#c084fc',
  '#f97316',
]

const MAX_NAME = 100
const MAX_CLASSES = 200
const FALLBACK_COLOR = '#64748b'

/** One editable row. `classId` is absent for a class that has not been saved yet. */
interface DraftClass {
  key: string
  classId?: string
  name: string
  description: string
  color: string
}

const norm = (s: string) => s.trim().replace(/\s+/g, ' ')
const fold = (s: string) => norm(s).toLowerCase()

let newKeyCounter = 0
const newKey = () => `new-${++newKeyCounter}`

const fromSaved = (c: LabelClass): DraftClass => ({
  key: c.classId,
  classId: c.classId,
  name: c.name,
  description: c.description ?? '',
  color: c.uiColorHex ?? FALLBACK_COLOR,
})

function nextColor(classes: DraftClass[]): string {
  const used = new Set(classes.map((c) => c.color.toLowerCase()))
  return CLASS_PALETTE.find((c) => !used.has(c)) ?? CLASS_PALETTE[classes.length % CLASS_PALETTE.length]
}

/** Why a name can't be used, or null when it is fine. A row may keep its own name. */
function nameError(name: string, draft: DraftClass[], ignoreKey?: string): string | null {
  const n = norm(name)
  if (!n) return 'Name is required'
  if (n.length > MAX_NAME) return `At most ${MAX_NAME} characters`
  if (draft.some((c) => c.key !== ignoreKey && fold(c.name) === fold(n))) return 'Already exists'
  return null
}

function parseBulk(text: string, draft: DraftClass[]) {
  const seen = new Set(draft.map((c) => fold(c.name)))
  const names: string[] = []
  const skipped: string[] = []
  for (const raw of text.split(/[\n,;]+/)) {
    const n = norm(raw)
    if (!n) continue
    if (n.length > MAX_NAME || seen.has(fold(n))) {
      skipped.push(n)
      continue
    }
    seen.add(fold(n))
    names.push(n)
  }
  return { names, skipped }
}

const same = (a: DraftClass, b: DraftClass) =>
  a.key === b.key &&
  a.name === b.name &&
  a.description === b.description &&
  a.color.toLowerCase() === b.color.toLowerCase()

function ClassManager({
  projectId,
  saved,
  counts,
  onSaved,
}: {
  projectId: string
  saved: LabelClass[]
  /** Samples per class id in the draft dataset; undefined until the health report has loaded. */
  counts: Map<string, number> | undefined
  onSaved: () => void
}) {
  const baseline = useMemo(() => saved.map(fromSaved), [saved])
  const [draft, setDraft] = useState<DraftClass[]>(baseline)
  const [newName, setNewName] = useState('')
  const [bulkOpen, setBulkOpen] = useState(false)
  const [bulkText, setBulkText] = useState('')

  const dirty = draft.length !== baseline.length || draft.some((c, i) => !same(c, baseline[i]))
  const errors = new Map(draft.map((c) => [c.key, nameError(c.name, draft, c.key)]))
  const hasErrors = [...errors.values()].some(Boolean)
  const max = Math.max(1, ...(counts?.values() ?? []))

  const diff = useMemo(() => {
    const byId = new Map(baseline.map((c) => [c.key, c]))
    const draftKeys = new Set(draft.map((c) => c.key))
    const kept = draft.filter((c) => byId.has(c.key))
    return {
      added: draft.filter((c) => !byId.has(c.key)).length,
      removed: baseline.filter((c) => !draftKeys.has(c.key)).length,
      renamed: kept.filter((c) => c.name !== byId.get(c.key)?.name).length,
    }
  }, [draft, baseline])

  const save = useMutation({
    mutationFn: () =>
      saveClasses(projectId, {
        classes: draft.map((c) => ({
          ...(c.classId && { classId: c.classId }),
          name: norm(c.name),
          description: c.description.trim() || null,
          uiColorHex: c.color,
        })),
      }),
    onSuccess: () => {
      onSaved()
      notifications.show({
        title: 'Classes saved',
        message: 'New snapshots will use this list. Existing snapshots are unchanged.',
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

  const patch = (key: string, change: Partial<DraftClass>) =>
    setDraft((d) => d.map((c) => (c.key === key ? { ...c, ...change } : c)))

  const addNames = (names: string[]) =>
    setDraft((d) => {
      let out = d
      for (const name of names) {
        if (out.length >= MAX_CLASSES) break
        out = [...out, { key: newKey(), name, description: '', color: nextColor(out) }]
      }
      return out
    })

  const addError = newName.trim() ? nameError(newName, draft) : null
  const submitNew = () => {
    const n = norm(newName)
    if (!n || addError) return
    addNames([n])
    setNewName('')
  }

  const remove = (row: DraftClass) => {
    const n = row.classId ? (counts?.get(row.classId) ?? 0) : 0
    const drop = () => setDraft((d) => d.filter((c) => c.key !== row.key))
    if (n === 0) return drop()
    modals.openConfirmModal({
      title: `Remove "${row.name}"?`,
      children: (
        <Text size="sm">
          {n.toLocaleString()} items in the draft are labelled "{row.name}". Existing snapshots keep the class; new
          snapshots will no longer include it, and those items will be unlabelled. Nothing changes until you save.
        </Text>
      ),
      labels: { confirm: 'Remove class', cancel: 'Keep' },
      confirmProps: { color: 'red' },
      onConfirm: drop,
    })
  }

  const bulk = parseBulk(bulkText, draft)

  return (
    <div className="flex flex-col gap-3">
      {hasErrors && (
        <Alert color="yellow" icon={<WarningCircleIcon size={16} />} p="xs">
          Fix the highlighted names before saving.
        </Alert>
      )}

      <Paper p="sm">
        <Group gap="xs" align="flex-start" wrap="nowrap">
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
          />
          <Button
            size="xs"
            leftSection={<PlusIcon size={14} />}
            onClick={submitNew}
            disabled={!newName.trim() || !!addError}
          >
            Add
          </Button>
          <Button
            size="xs"
            variant="default"
            onClick={() => {
              setBulkText('')
              setBulkOpen(true)
            }}
          >
            Add several…
          </Button>
        </Group>
      </Paper>

      <Table withTableBorder verticalSpacing={6} highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th w={150}>Colour</Table.Th>
            <Table.Th w={220}>Name</Table.Th>
            <Table.Th>Description</Table.Th>
            <Table.Th w={240}>Samples in draft</Table.Th>
            <Table.Th w={50} />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {draft.map((row, i) => {
            const n = row.classId ? counts?.get(row.classId) : undefined
            return (
              <Table.Tr key={row.key}>
                <Table.Td>
                  <ColorInput
                    size="xs"
                    value={row.color}
                    format="hex"
                    swatches={CLASS_PALETTE}
                    withEyeDropper={false}
                    popoverProps={{ withinPortal: true }}
                    aria-label={`Colour of ${row.name}`}
                    onChange={(v) => patch(row.key, { color: v })}
                  />
                </Table.Td>
                <Table.Td>
                  <TextInput
                    size="xs"
                    value={row.name}
                    error={errors.get(row.key)}
                    aria-label={`Name of class ${i + 1}`}
                    onChange={(e) => patch(row.key, { name: e.currentTarget.value })}
                    onBlur={() => !errors.get(row.key) && patch(row.key, { name: norm(row.name) })}
                  />
                </Table.Td>
                <Table.Td>
                  <TextInput
                    size="xs"
                    value={row.description}
                    placeholder="Description (optional)"
                    aria-label={`Description of ${row.name}`}
                    onChange={(e) => patch(row.key, { description: e.currentTarget.value })}
                  />
                </Table.Td>
                <Table.Td>
                  {n === undefined ? (
                    <Text size="xs" c="dimmed">
                      {row.classId ? '—' : 'new'}
                    </Text>
                  ) : (
                    <Group gap="xs" wrap="nowrap">
                      <MeterBar value={n} max={max} color={row.color} height={8} />
                      <Text size="xs" className="tnum" w={56} ta="right">
                        {n.toLocaleString()}
                      </Text>
                    </Group>
                  )}
                </Table.Td>
                <Table.Td>
                  <ActionIcon
                    color="red"
                    variant="subtle"
                    aria-label={`Remove ${row.name}`}
                    onClick={() => remove(row)}
                  >
                    <TrashIcon size={15} />
                  </ActionIcon>
                </Table.Td>
              </Table.Tr>
            )
          })}
          {draft.length === 0 && (
            <Table.Tr>
              <Table.Td colSpan={5}>
                <Text ta="center" c="dimmed" py="md">
                  No classes yet. Add the labels your model should predict.
                </Text>
              </Table.Td>
            </Table.Tr>
          )}
        </Table.Tbody>
      </Table>

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
          </Group>
          <Group gap="xs">
            <Button variant="default" size="compact-md" disabled={!dirty} onClick={() => setDraft(baseline)}>
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
        <Text size="xs" c="dimmed" mt={4}>
          Class edits apply to <b>new</b> snapshots and runs. Existing snapshots are immutable and keep the classes they
          were built with.
        </Text>
      </Paper>

      <Modal opened={bulkOpen} onClose={() => setBulkOpen(false)} title="Add several classes" centered>
        <Stack>
          <Textarea
            autosize
            minRows={5}
            maxRows={10}
            placeholder="One per line, or comma-separated"
            value={bulkText}
            onChange={(e) => setBulkText(e.currentTarget.value)}
            data-autofocus
          />
          <Text size="xs" c="dimmed">
            {bulk.names.length} to add
            {bulk.skipped.length ? ` · skipped (duplicate or too long): ${bulk.skipped.join(', ')}` : ''}
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setBulkOpen(false)}>
              Cancel
            </Button>
            <Button
              disabled={bulk.names.length === 0}
              onClick={() => {
                addNames(bulk.names)
                setBulkOpen(false)
              }}
            >
              Add {bulk.names.length}
            </Button>
          </Group>
        </Stack>
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

export function ClassesPage() {
  const { projectId } = routeApi.useParams()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(project.task)
  const usesClasses = descriptor.annotation.requiresLabelClasses
  const { data, isLoading, isError, refetch } = useLabelClasses(usesClasses ? projectId : undefined)
  const saved = data?.classes ?? []

  // Sample counts come from the draft's health report (the same numbers the Dataset page shows).
  const { data: healthData } = useDatasetHealth(projectId, usesClasses)
  const counts = useMemo(
    () => (healthData ? new Map(healthData.health.classDistribution.map((c) => [c.classId, c.count])) : undefined),
    [healthData],
  )

  return (
    <div className="flex flex-col gap-3 p-3" style={{ maxWidth: 980 }}>
      <PageHeader
        title="Classes"
        description={`The labels this project's ${descriptor.label.toLowerCase()} model predicts.`}
      />

      {!usesClasses ? (
        <Alert icon={<InfoIcon size={16} />} color="blue" title="This task has no classes">
          {descriptor.label} predicts a value rather than a label from a fixed set, so there is nothing to manage here.
        </Alert>
      ) : (
        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {/* key: a saved change from the server resets the editable draft */}
          <ClassManager
            key={saved.map((c) => `${c.classId}:${c.name}:${c.description}:${c.uiColorHex}`).join('|')}
            projectId={projectId}
            saved={saved}
            counts={counts}
            onSaved={() => invalidateProjectScope(queryClient, projectId)}
          />
        </QueryBoundary>
      )}
    </div>
  )
}

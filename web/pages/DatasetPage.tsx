/**
 * Dataset page – the mutable draft: browse pool items, assign them to
 * splits (manually or with one-click auto-split), assign label classes in
 * bulk, and freeze the current state into an immutable snapshot. Snapshots
 * themselves are browsed on the Snapshots page — this page only ever shows
 * the draft.
 */

import { Button, Card, Group, Menu, Modal, Paper, Stack, Tabs, Text, Tooltip } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import {
  CaretDownIcon,
  DatabaseIcon,
  HeartbeatIcon,
  ListChecksIcon,
  PlusIcon,
  ShuffleIcon,
  TagIcon,
  TrashIcon,
  XIcon,
} from '@phosphor-icons/react'
import { AutoSplitForm } from '@public/components/dataset/AutoSplitForm'
import { DatasetHealthPanel } from '@public/components/dataset/DatasetHealthPanel'
import { ItemsByModality } from '@public/components/dataset/ItemsByModality'
import { ItemsFilterBar } from '@public/components/dataset/ItemsFilterBar'
import { ItemsPaginationBar } from '@public/components/dataset/ItemsPaginationBar'
import { SPLIT_TYPES, SplitProgressBar, splitCounts } from '@public/components/dataset/VersionBrowsing'
import { confirmDelete, EmptyState, LinkButton, PageHeader } from '@public/components/ui'
import {
  classifyItems,
  createAnnotation,
  deleteItems,
  setItemsSplit,
  updateAnnotation,
} from '@public/lib/api/generated/datasets/datasets'
import {
  invalidateProjectScope,
  projectDetailQueryOptions,
  projectItemsQueryOptions,
  useLabelClasses,
  useProjectItems,
} from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useEffect, useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/dataset')

/* ── Bulk actions toolbar ── */
const BulkActionsToolbar = ({
  projectId,
  selectedIds,
  onClear,
  requiresLabelClasses,
  pageCount,
  total,
  onSelectPage,
  onSelectAllMatching,
  selectingAll,
}: {
  projectId: string
  selectedIds: Set<string>
  onClear: () => void
  requiresLabelClasses: boolean
  pageCount: number
  total: number
  onSelectPage: () => void
  onSelectAllMatching: () => void
  selectingAll: boolean
}) => {
  const queryClient = useQueryClient()
  const { data: classesData } = useLabelClasses(requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []

  // Project scope, not just the items query: reassigning splits and assigning
  // classes both change counts that are rendered from project detail.
  const invalidate = () => invalidateProjectScope(queryClient, projectId)

  const reassignSplit = useMutation({
    mutationFn: async (split: (typeof SPLIT_TYPES)[number]) => {
      await setItemsSplit(projectId, { itemIds: [...selectedIds], split })
    },
    onSuccess: () => {
      notifications.show({ title: 'Split updated', message: `${selectedIds.size} item(s) reassigned`, color: 'green' })
      invalidate()
      onClear()
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to reassign split', color: 'red' }),
  })

  const assignClass = useMutation({
    mutationFn: async (classId: string) => {
      await classifyItems(projectId, { itemIds: [...selectedIds], classId })
    },
    onSuccess: () => {
      notifications.show({ title: 'Class assigned', message: `${selectedIds.size} item(s) updated`, color: 'green' })
      invalidate()
      onClear()
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to assign class', color: 'red' }),
  })

  const bulkDelete = useMutation({
    mutationFn: async () => {
      await deleteItems(projectId, { itemIds: [...selectedIds] })
    },
    onSuccess: () => {
      notifications.show({
        title: 'Deleted',
        message: `${selectedIds.size} item(s) removed from the draft`,
        color: 'green',
      })
      invalidate()
      onClear()
    },
    onError: () => notifications.show({ title: 'Error', message: 'Bulk delete failed', color: 'red' }),
  })

  const handleDelete = () =>
    confirmDelete({
      title: 'Delete items',
      message: `Remove ${selectedIds.size} item(s) from the draft? Items still referenced by a snapshot are kept (hidden from the draft, preserved for that snapshot).`,
      onConfirm: () => bulkDelete.mutate(),
    })

  return (
    <Card
      withBorder
      p="xs"
      radius="md"
      shadow="lg"
      style={{
        position: 'sticky',
        bottom: '4rem',
        zIndex: 200,
      }}
    >
      <Group justify="space-between">
        <Text size="sm" fw={600}>
          {selectedIds.size} selected
        </Text>
        <Group gap="xs">
          <Menu shadow="md" position="bottom-start">
            <Menu.Target>
              <Button
                size="xs"
                variant="light"
                color="gray"
                rightSection={<CaretDownIcon size={12} />}
                leftSection={<ListChecksIcon size={12} />}
                loading={selectingAll}
              >
                Select
              </Button>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item onClick={onSelectPage}>Select all on this page ({pageCount})</Menu.Item>
              <Menu.Item onClick={onSelectAllMatching}>Select all matching items ({total})</Menu.Item>
            </Menu.Dropdown>
          </Menu>

          {selectedIds.size > 0 && (
            <>
              <Menu shadow="md" position="bottom-start">
                <Menu.Target>
                  <Button size="xs" variant="light" rightSection={<CaretDownIcon size={12} />}>
                    Assign split
                  </Button>
                </Menu.Target>
                <Menu.Dropdown>
                  {SPLIT_TYPES.map((s) => (
                    <Menu.Item key={s} tt="capitalize" onClick={() => reassignSplit.mutate(s)}>
                      {s}
                    </Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>

              {requiresLabelClasses && classes.length > 0 && (
                <Menu shadow="md" position="bottom-start">
                  <Menu.Target>
                    <Button
                      size="xs"
                      variant="light"
                      color="violet"
                      rightSection={<CaretDownIcon size={12} />}
                      leftSection={<TagIcon size={12} />}
                    >
                      Assign class
                    </Button>
                  </Menu.Target>
                  <Menu.Dropdown>
                    {classes.map((c) => (
                      <Menu.Item key={c.classId} onClick={() => assignClass.mutate(c.classId)}>
                        {c.name}
                      </Menu.Item>
                    ))}
                  </Menu.Dropdown>
                </Menu>
              )}

              <Button
                size="xs"
                variant="light"
                color="red"
                leftSection={<TrashIcon size={12} />}
                onClick={handleDelete}
              >
                Delete
              </Button>
            </>
          )}
          <Tooltip label="Clear selection">
            <Button size="xs" variant="subtle" color="gray" onClick={onClear} disabled={selectedIds.size === 0}>
              <XIcon size={14} />
            </Button>
          </Tooltip>
        </Group>
      </Group>
    </Card>
  )
}

/* ── Main Dataset page ── */
export function DatasetPage() {
  const { projectId } = routeApi.useParams()
  const { page, perPage, split, classId, search, sort, tab = 'browse' } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const dataset = activeProject.dataset
  const descriptor = getTaskDescriptor(activeProject.task)

  const [autoSplitOpened, { open: openAutoSplit, close: closeAutoSplit }] = useDisclosure(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const queryClient = useQueryClient()

  // Selection is scoped to the current split/class/search filter — changing
  // any of them invalidates any selection made under the previous filter.
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional reset-on-change, split/classId/search aren't read in the body
  useEffect(() => setSelectedIds(new Set()), [split, classId, search])

  const toggleSelect = (itemId: string) =>
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(itemId)) next.delete(itemId)
      else next.add(itemId)
      return next
    })

  const toggleSelectMany = (itemIds: string[], selected: boolean) =>
    setSelectedIds((prev) => {
      const next = new Set(prev)
      for (const id of itemIds) {
        if (selected) next.add(id)
        else next.delete(id)
      }
      return next
    })

  const { data: classesData } = useLabelClasses(descriptor.annotation.requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []
  const { data, isLoading, isError, refetch } = useProjectItems(dataset?.draft ? projectId : undefined, {
    versionId: dataset?.draft?.id,
    split: split ?? undefined,
    classId: classId ?? undefined,
    search: search ?? undefined,
    page,
    perPage,
    sort: sort ?? undefined,
  })
  const items = data?.items ?? []
  const total = data?.total ?? 0
  const classNameById = new Map(classes.map((c) => [c.classId, c.name]))
  const classCounts: Record<string, number> = { unassigned: data?.unassignedCount ?? 0 }
  for (const c of data?.classCounts ?? []) classCounts[c.classId] = c.count

  // Bulk mutations cap at 1000 item ids per request (server-side), so
  // "select all matching items" fetches at most that many even if the
  // filtered total is larger.
  const BULK_SELECTION_CAP = 1000
  const selectAllMatching = useMutation({
    mutationFn: async () => {
      const matchingPerPage = Math.min(total, BULK_SELECTION_CAP)
      const result = await queryClient.fetchQuery(
        projectItemsQueryOptions(projectId, {
          versionId: dataset?.draft?.id,
          split: split ?? undefined,
          classId: classId ?? undefined,
          search: search ?? undefined,
          page: 1,
          perPage: matchingPerPage,
          sort: sort ?? undefined,
        }),
      )
      return result.items.map((item) => item.id)
    },
    onSuccess: (ids) => {
      setSelectedIds(new Set(ids))
      if (total > BULK_SELECTION_CAP) {
        notifications.show({
          title: 'Selection capped',
          message: `Selected the first ${BULK_SELECTION_CAP} of ${total} matching items — bulk actions support at most ${BULK_SELECTION_CAP} at a time.`,
          color: 'yellow',
        })
      }
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to select all matching items', color: 'red' }),
  })

  // Free-text ground truth (captioning/ASR) has no shared value to bulk-assign
  // the way a class does, so it's edited per-item from the detail modal
  // instead — see ItemsByModality.tsx's CaptionEditor.
  const saveCaption = useMutation({
    mutationFn: async ({
      itemId,
      existingAnnotationId,
      text,
    }: {
      itemId: string
      existingAnnotationId: string | null
      text: string
    }) => {
      if (existingAnnotationId) await updateAnnotation(existingAnnotationId, { labelTextSequence: text })
      else await createAnnotation(itemId, { annotationType: 'text_sequence', labelTextSequence: text })
    },
    onSuccess: () => invalidateProjectScope(queryClient, projectId),
    onError: () => notifications.show({ title: 'Error', message: 'Failed to save', color: 'red' }),
  })
  const captionLabel =
    activeProject.task === 'automatic_speech_recognition'
      ? 'Transcript'
      : descriptor.annotation.type === 'text_sequence'
        ? 'Caption'
        : null
  const captionEditing = captionLabel
    ? {
        label: captionLabel,
        onSave: async (itemId: string, existingAnnotationId: string | null, text: string) => {
          await saveCaption.mutateAsync({ itemId, existingAnnotationId, text })
        },
      }
    : undefined

  const draftSplitCounts = dataset?.draft ? splitCounts(dataset.draft) : null

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Dataset"
        description={
          draftSplitCounts
            ? `${dataset?.draft?.itemCount ?? 0} items · train ${draftSplitCounts.train} · validation ${draftSplitCounts.validation} · test ${draftSplitCounts.test}`
            : undefined
        }
        actions={
          dataset?.draft && (
            <>
              <Button
                size="compact-md"
                leftSection={<ShuffleIcon size={14} />}
                variant="default"
                onClick={openAutoSplit}
              >
                Auto-split
              </Button>
              <LinkButton
                to="/project/$projectId/snapshots/new"
                params={{ projectId }}
                leftSection={<PlusIcon size={14} />}
              >
                Create snapshot
              </LinkButton>
            </>
          )
        }
      />

      {!dataset?.draft ? (
        <EmptyState
          icon={DatabaseIcon}
          title="No draft dataset found"
          description="This project's dataset hasn't been initialized yet."
        />
      ) : (
        <Tabs
          value={tab}
          onChange={(t) =>
            navigate({ search: (prev) => ({ ...prev, tab: t === 'browse' ? undefined : (t as 'distribution') }) })
          }
        >
          <Tabs.List>
            <Tabs.Tab value="browse">Browse</Tabs.Tab>
            <Tabs.Tab value="distribution" leftSection={<HeartbeatIcon size={14} />}>
              Distribution
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="browse" pt="sm">
            <Stack gap="sm">
              <SplitProgressBar version={dataset.draft} />
              <ItemsFilterBar
                version={dataset.draft}
                split={split ?? null}
                onSplitChange={(s) => navigate({ search: (prev) => ({ ...prev, split: s ?? undefined, page: 1 }) })}
                classes={classes}
                classId={classId ?? null}
                onClassChange={(c) => navigate({ search: (prev) => ({ ...prev, classId: c ?? undefined, page: 1 }) })}
                classCounts={classCounts}
                search={search ?? ''}
                onSearchChange={(s) => navigate({ search: (prev) => ({ ...prev, search: s || undefined, page: 1 }) })}
                sort={sort ?? 'newest'}
                onSortChange={(sort) => navigate({ search: (prev) => ({ ...prev, sort, page: 1 }) })}
              />

              <ItemsByModality
                items={items}
                modality={dataset.modality}
                isLoading={isLoading}
                isError={isError}
                onRetry={() => refetch()}
                classNameById={classNameById}
                emptyMessage="No items in the draft yet — add some from the Upload page."
                selection={{ selectedIds, onToggle: toggleSelect, onToggleMany: toggleSelectMany }}
                captionEditing={captionEditing}
              />

              {items.length > 0 && (
                <BulkActionsToolbar
                  projectId={projectId}
                  selectedIds={selectedIds}
                  onClear={() => setSelectedIds(new Set())}
                  requiresLabelClasses={descriptor.annotation.requiresLabelClasses}
                  pageCount={items.length}
                  total={total}
                  onSelectPage={() => setSelectedIds(new Set(items.map((i) => i.id)))}
                  onSelectAllMatching={() => selectAllMatching.mutate()}
                  selectingAll={selectAllMatching.isPending}
                />
              )}

              {items.length > 0 && (
                <ItemsPaginationBar
                  total={total}
                  page={page}
                  perPage={perPage}
                  onPageChange={(page) => navigate({ search: (prev) => ({ ...prev, page }) })}
                  onPerPageChange={(perPage) => navigate({ search: (prev) => ({ ...prev, perPage, page: 1 }) })}
                />
              )}
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value="distribution" pt="sm">
            <Paper p="md">
              <DatasetHealthPanel projectId={projectId} active={tab === 'distribution'} />
            </Paper>
          </Tabs.Panel>
        </Tabs>
      )}

      <Modal opened={autoSplitOpened} onClose={closeAutoSplit} title="Auto-split the draft" centered>
        <AutoSplitForm
          projectId={projectId}
          requiresLabelClasses={descriptor.annotation.requiresLabelClasses}
          onDone={closeAutoSplit}
          onCancel={closeAutoSplit}
        />
      </Modal>
    </div>
  )
}

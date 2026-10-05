/**
 * Dataset page – the mutable draft: browse pool items, assign them to
 * splits (manually or with one-click auto-split), assign label classes in
 * bulk, and freeze the current state into an immutable snapshot. Snapshots
 * themselves are browsed on the Snapshots page — this page only ever shows
 * the draft.
 */

import { Button, Modal, Paper, Stack, Tabs } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { DatabaseIcon, HeartbeatIcon, PlusIcon, ShuffleIcon } from '@phosphor-icons/react'
import { AutoSplitForm } from '@public/components/dataset/AutoSplitForm'
import { BulkActionsToolbar } from '@public/components/dataset/BulkActionsToolbar'
import { DatasetHealthPanel } from '@public/components/dataset/DatasetHealthPanel'
import { type ItemEditing, ItemsByModality } from '@public/components/dataset/ItemsByModality'
import { type ItemLabeled, ItemsFilterBar } from '@public/components/dataset/ItemsFilterBar'
import { ItemsPaginationBar } from '@public/components/dataset/ItemsPaginationBar'
import { SplitProgressBar, splitCounts } from '@public/components/dataset/VersionBrowsing'
import { EmptyState, LinkButton, PageHeader } from '@public/components/ui'
import { useItemMutations } from '@public/hooks/useItemMutations'
import { useItemSelection } from '@public/hooks/useItemSelection'
import {
  projectDetailQueryOptions,
  projectItemsQueryOptions,
  useLabelClasses,
  useProjectItems,
} from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/dataset')

// Bulk mutations cap at 1000 item ids per request (server-side), so
// "select all matching items" fetches at most that many even if the
// filtered total is larger.
const BULK_SELECTION_CAP = 1000

/* ── Main Dataset page ── */
export function DatasetPage() {
  const { projectId } = routeApi.useParams()
  const { page, perPage, split, classId, search, sort, labeled, tab = 'browse' } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const dataset = activeProject.dataset
  const descriptor = getTaskDescriptor(activeProject.task)

  const [autoSplitOpened, { open: openAutoSplit, close: closeAutoSplit }] = useDisclosure(false)
  const queryClient = useQueryClient()

  const { data: classesData } = useLabelClasses(descriptor.annotation.requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []

  // Everything that selects *which* items, minus paging. Used for both the
  // visible page and "select all matching", so the two can't drift apart.
  const filters = {
    versionId: dataset?.draft?.id,
    split: split ?? undefined,
    classId: classId ?? undefined,
    search: search ?? undefined,
    sort: sort ?? undefined,
    labeled: labeled === undefined ? undefined : labeled === 'labeled',
  }
  const { data, isLoading, isError, refetch } = useProjectItems(dataset?.draft ? projectId : undefined, {
    ...filters,
    page,
    perPage,
  })
  const items = data?.items ?? []
  const total = data?.total ?? 0
  const classNameById = new Map(classes.map((c) => [c.classId, c.name]))
  const classCounts: Record<string, number> = { unassigned: data?.unassignedCount ?? 0 }
  for (const c of data?.classCounts ?? []) classCounts[c.classId] = c.count

  // Selection is scoped to the current split/class/search filter — changing
  // any of them invalidates any selection made under the previous filter.
  const { selection, selectedIds, clear, replace } = useItemSelection({
    visibleIds: items.map((i) => i.id),
    resetKey: `${split ?? ''}|${classId ?? ''}|${search ?? ''}|${labeled ?? ''}`,
  })

  const selectAllMatching = useMutation({
    mutationFn: async () => {
      const result = await queryClient.fetchQuery(
        projectItemsQueryOptions(projectId, { ...filters, page: 1, perPage: Math.min(total, BULK_SELECTION_CAP) }),
      )
      return result.items.map((item) => item.id)
    },
    onSuccess: (ids) => {
      replace(ids)
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

  // Free-text ground truth (captioning/ASR/generation) is edited per-item from the detail
  // drawer instead of in bulk — see items/CaptionEditor.tsx.
  const { saveCaption, saveTarget, reassignSplit, assignClass, removeItems } = useItemMutations(projectId)
  const outputColumn = descriptor.output?.column ?? 'text'
  const captionLabel =
    descriptor.annotation.type !== 'text_sequence'
      ? null
      : activeProject.task === 'automatic_speech_recognition'
        ? 'Transcript'
        : descriptor.itemSpec.payload === 'file'
          ? 'Caption'
          : // An inline-text task labels with its output column ("completion", "summary", "answer", ...).
            outputColumn.charAt(0).toUpperCase() + outputColumn.slice(1)
  const captionEditing = captionLabel
    ? {
        label: captionLabel,
        onSave: async (itemId: string, existingAnnotationId: string | null, text: string) => {
          await saveCaption.mutateAsync({ itemId, existingAnnotationId, text })
        },
      }
    : undefined

  // Everything the drawer and the per-item menu may change. Each is the one-item case of a bulk mutation,
  // so they notify and invalidate exactly as the bulk toolbar does.
  const editing: ItemEditing = {
    classes: descriptor.annotation.requiresLabelClasses ? classes : [],
    onSetSplit: (itemId, split) => reassignSplit.mutate({ itemIds: [itemId], split }),
    onSetClass: (itemId, classId) => assignClass.mutate({ itemIds: [itemId], classId }),
    onSaveTarget: async (itemId, existingAnnotationId, value) => {
      await saveTarget.mutateAsync({ itemId, existingAnnotationId, value })
    },
    onDelete: (itemId) => removeItems.mutate({ itemIds: [itemId] }),
  }

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
                // Class tasks filter by class (incl. "Unassigned"); the rest have no class to filter on.
                {...(descriptor.annotation.requiresLabelClasses
                  ? {}
                  : {
                      labeled: labeled ?? null,
                      onLabeledChange: (l: ItemLabeled | null) =>
                        navigate({ search: (prev) => ({ ...prev, labeled: l ?? undefined, page: 1 }) }),
                    })}
              />

              <ItemsByModality
                items={items}
                modality={dataset.modality}
                annotation={descriptor.annotation}
                isLoading={isLoading}
                isError={isError}
                onRetry={() => refetch()}
                classNameById={classNameById}
                emptyMessage="No items in the draft yet — add some from the Upload page."
                selection={selection}
                captionEditing={captionEditing}
                editing={editing}
              />

              {items.length > 0 && (
                <BulkActionsToolbar
                  projectId={projectId}
                  selectedIds={selectedIds}
                  onClear={clear}
                  requiresLabelClasses={descriptor.annotation.requiresLabelClasses}
                  pageCount={items.length}
                  total={total}
                  onSelectPage={() => replace(items.map((i) => i.id))}
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

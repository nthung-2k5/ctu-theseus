/**
 * Data page – everything about the mutable draft in one place: browse pool items and assign them to splits
 * (manually or with one-click auto-split) and classes (in bulk), manage the label classes, check the
 * distribution, add more data from a drawer, and freeze the current state into an immutable snapshot. Snapshots
 * themselves are browsed on the Snapshots page — this page only ever shows the draft.
 */

import { Button, Drawer, Modal, Paper, Stack, Tabs } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { ArchiveIcon, DatabaseIcon, HeartbeatIcon, ShuffleIcon, TagIcon, UploadSimpleIcon } from '@phosphor-icons/react'
import { ClassesPanel } from '@public/components/classes/ClassesPanel'
import { AddDataPanel } from '@public/components/dataset/AddDataPanel'
import { AutoSplitForm } from '@public/components/dataset/AutoSplitForm'
import { BulkActionsToolbar } from '@public/components/dataset/BulkActionsToolbar'
import { DatasetHealthPanel } from '@public/components/dataset/DatasetHealthPanel'
import { type ItemEditing, ItemsByModality } from '@public/components/dataset/ItemsByModality'
import { type ItemLabeled, ItemsFilterBar } from '@public/components/dataset/ItemsFilterBar'
import { ItemsFooter } from '@public/components/dataset/ItemsFooter'
import { SplitProgressBar, splitCounts } from '@public/components/dataset/VersionBrowsing'
import { EmptyState, LinkButton, PageHeader } from '@public/components/ui'
import { oneItem, useItemMutations } from '@public/hooks/useItemMutations'
import { useItemSelection } from '@public/hooks/useItemSelection'
import type { MatchFilter } from '@public/lib/itemSelection'
import { projectDetailQueryOptions, useLabelClasses, useProjectItems } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/data')

type DataTab = 'browse' | 'classes' | 'distribution'

/* ── Main Data page ── */
export function DataPage() {
  const { projectId } = routeApi.useParams()
  const { page, perPage, split, classId, search, sort, labeled, tab: requestedTab, add } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const dataset = activeProject.dataset
  const descriptor = getTaskDescriptor(activeProject.task)

  const [autoSplitOpened, { open: openAutoSplit, close: closeAutoSplit }] = useDisclosure(false)
  const hasClasses = descriptor.annotation.requiresLabelClasses
  // A link to the Classes tab of a task that has none falls back to Browse.
  const tab: DataTab = requestedTab === 'classes' && !hasClasses ? 'browse' : (requestedTab ?? 'browse')
  const addOpened = add === true
  const setAddOpened = (open: boolean) =>
    navigate({ search: (prev) => ({ ...prev, add: open ? (true as const) : undefined }), replace: true })

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
  const classColorById = new Map(classes.flatMap((c) => (c.uiColorHex ? [[c.classId, c.uiColorHex] as const] : [])))
  // Boxes carry a class EACH, drawn on the image, so there is no whole-item class to assign in bulk or from a menu.
  const labelsWithBoxes = descriptor.annotation.type === 'bounding_box'
  const classCounts: Record<string, number> = { unassigned: data?.unassignedCount ?? 0 }
  for (const c of data?.classCounts ?? []) classCounts[c.classId] = c.count

  // Selection is scoped to the current split/class/search/labeled filter — changing any of them
  // invalidates any selection made under the previous filter. It survives paging, and "select all
  // matching" is a filter the server resolves, not a list of ids fetched here.
  const matchFilter: MatchFilter = {
    split: filters.split,
    classId: filters.classId,
    search: filters.search,
    labeled: filters.labeled,
  }
  const bulk = useItemSelection({
    visibleIds: items.map((i) => i.id),
    resetKey: `${split ?? ''}|${classId ?? ''}|${search ?? ''}|${labeled ?? ''}`,
  })

  // Free-text ground truth (captioning/ASR/generation) is edited per-item from the detail
  // drawer instead of in bulk — see items/CaptionEditor.tsx.
  const { saveCaption, saveTarget, reassignSplit, assignClass, removeItems, addBox, updateBox, removeBox, markEmpty } =
    useItemMutations(projectId)
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
    classes: descriptor.annotation.requiresLabelClasses && !labelsWithBoxes ? classes : [],
    onSetSplit: (itemId, split) => reassignSplit.mutate({ selection: oneItem(itemId), split }),
    onSetClass: (itemId, classId) => assignClass.mutate({ selection: oneItem(itemId), classId }),
    onSaveTarget: async (itemId, existingAnnotationId, value) => {
      await saveTarget.mutateAsync({ itemId, existingAnnotationId, value })
    },
    onDelete: (itemId) => removeItems.mutate({ selection: oneItem(itemId) }),
    boxes: labelsWithBoxes
      ? {
          classes,
          onAdd: async (itemId, classId, box) => {
            await addBox.mutateAsync({ itemId, classId, box })
          },
          onUpdate: async (annotationId, patch) => {
            await updateBox.mutateAsync({ annotationId, ...patch })
          },
          onRemove: async (annotationId) => {
            await removeBox.mutateAsync({ annotationId })
          },
          onMarkEmpty: async (itemId) => {
            await markEmpty.mutateAsync({ itemId })
          },
        }
      : undefined,
  }

  const draftSplitCounts = dataset?.draft ? splitCounts(dataset.draft) : null
  const isEmptyDraft = (dataset?.draft?.itemCount ?? 0) === 0

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Data"
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
                disabled={isEmptyDraft}
              >
                Auto-split
              </Button>
              <LinkButton
                to="/project/$projectId/snapshots/new"
                params={{ projectId }}
                variant="default"
                leftSection={<ArchiveIcon size={14} />}
                disabled={isEmptyDraft}
              >
                Create snapshot
              </LinkButton>
              <Button leftSection={<UploadSimpleIcon size={14} />} onClick={() => setAddOpened(true)}>
                Add data
              </Button>
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
            navigate({ search: (prev) => ({ ...prev, tab: t === 'browse' ? undefined : (t as DataTab) }) })
          }
        >
          <Tabs.List>
            <Tabs.Tab value="browse" leftSection={<DatabaseIcon size={14} />}>
              Browse
            </Tabs.Tab>
            {hasClasses && (
              <Tabs.Tab value="classes" leftSection={<TagIcon size={14} />}>
                Classes
              </Tabs.Tab>
            )}
            <Tabs.Tab value="distribution" leftSection={<HeartbeatIcon size={14} />}>
              Distribution
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="browse" pt="sm">
            {isEmptyDraft ? (
              <EmptyState
                icon={UploadSimpleIcon}
                title="No data yet"
                description={
                  hasClasses
                    ? 'Add the files you want to train on. Define your classes in the Classes tab first, or assign them afterwards.'
                    : 'Add the files or rows you want to train on.'
                }
                action={
                  <Button leftSection={<UploadSimpleIcon size={14} />} onClick={() => setAddOpened(true)}>
                    Add data
                  </Button>
                }
              />
            ) : (
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
                  classColorById={classColorById}
                  emptyMessage="No items match these filters."
                  selection={bulk.selection}
                  captionEditing={captionEditing}
                  editing={editing}
                />

                {items.length > 0 && (
                  <ItemsFooter
                    bulk={
                      <BulkActionsToolbar
                        projectId={projectId}
                        request={bulk.request(matchFilter)}
                        count={bulk.count(total)}
                        exclusionCount={bulk.exclusionCount}
                        onClear={bulk.clear}
                        requiresLabelClasses={descriptor.annotation.requiresLabelClasses && !labelsWithBoxes}
                        pageCount={items.length}
                        total={total}
                        onSelectPage={() => bulk.selectPage(items.map((i) => i.id))}
                        onSelectAllMatching={bulk.selectAllMatching}
                      />
                    }
                    total={total}
                    page={page}
                    perPage={perPage}
                    onPageChange={(page) => navigate({ search: (prev) => ({ ...prev, page }) })}
                    onPerPageChange={(perPage) => navigate({ search: (prev) => ({ ...prev, perPage, page: 1 }) })}
                  />
                )}
              </Stack>
            )}
          </Tabs.Panel>

          {hasClasses && (
            <Tabs.Panel value="classes" pt="sm">
              {tab === 'classes' && <ClassesPanel projectId={projectId} />}
            </Tabs.Panel>
          )}

          <Tabs.Panel value="distribution" pt="sm">
            <Paper p="md">
              <DatasetHealthPanel projectId={projectId} active={tab === 'distribution'} />
            </Paper>
          </Tabs.Panel>
        </Tabs>
      )}

      {/* Kept mounted while closed, so a half-staged upload survives closing the drawer. */}
      <Drawer
        opened={addOpened}
        onClose={() => setAddOpened(false)}
        position="right"
        size="min(1100px, 96vw)"
        title="Add data"
        keepMounted
        styles={{ body: { height: 'calc(100% - 60px)' } }}
      >
        <AddDataPanel projectId={projectId} />
      </Drawer>

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

/**
 * Add-data panel – gets raw data into the project. It opens in a drawer of the data page.
 *
 * One flow for every task: add → review → upload. Adding is task-shaped: vision/audio/video tasks drop files,
 * folders or archives onto a class / split folder tree (see FileImport), text tasks type items inline or import
 * many at once from .txt/.md files and CSV/TSV/JSONL rows, and tabular tasks import a CSV. Everything lands in the
 * same review (the folder tree or a flat table), under the same "Ready to upload" bar.
 *
 * Nothing is sent until the user presses Upload. A split left on "auto" is drawn when they do, and then the batch
 * becomes a background job (UploadTray) — the drawer can be closed and the next batch staged while it runs. Uploaded
 * items land in the mutable draft, ready for snapshotting. The drawer stays mounted while closed so a half-staged
 * batch survives closing it.
 */

import { Button, Group, Modal, Paper, Select, Stack, Tabs, Text, Textarea } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { CloudArrowUpIcon } from '@phosphor-icons/react'
import { FileImport } from '@public/components/dataset/FileImport'
import { TabularCsvImporter } from '@public/components/dataset/TabularCsvImporter'
import { TextBatchImporter } from '@public/components/dataset/TextBatchImporter'
import { YoloImporter } from '@public/components/dataset/YoloImporter'
import { SectionLabel } from '@public/components/ui'
import { STAGED_SPLIT_OPTIONS } from '@public/lib/constants'
import { projectDetailQueryOptions, useLabelClasses } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import type { CommitSettings } from '@public/lib/upload/commit'
import { classIdOfKey, existingKey } from '@public/lib/upload/fileTree'
import { itemToRow } from '@public/lib/upload/review'
import { DEFAULT_RATIOS } from '@public/lib/upload/splitPlan'
import type { StagedSplit } from '@public/lib/upload/types'
import { useFileStaging } from '@public/lib/upload/useFileStaging'
import { type StagedDraft, type StagedEdit, useUploadQueue } from '@public/lib/uploadQueue'
import type { LabelClass } from '@public/store/types'
import { useSuspenseQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { CommitBar, type CommitSummary } from './CommitBar'
import { ReviewTable, type RowEdit } from './ReviewTable'
import { useFileCommit, useItemCommit } from './useCommit'

/**
 * The text entry panel picks a split/class up front purely as the *default* for
 * whatever it stages next — the review table is where these get corrected,
 * so nothing here is a final assignment.
 */
function StagingDefaults({
  split,
  onSplitChange,
  classId,
  onClassChange,
  classes,
}: {
  split: StagedSplit
  onSplitChange: (split: StagedSplit) => void
  classId: string | null
  onClassChange: (classId: string | null) => void
  classes: LabelClass[]
}) {
  return (
    <Group grow align="flex-start">
      <Select
        label="Split"
        data={STAGED_SPLIT_OPTIONS}
        value={split}
        onChange={(v) => onSplitChange((v ?? 'auto') as StagedSplit)}
        allowDeselect={false}
      />
      {classes.length > 0 && (
        <Select
          label="Class"
          placeholder="None"
          data={classes.map((c) => ({ value: c.classId, label: c.name }))}
          value={classId}
          onChange={onClassChange}
          searchable
          clearable
        />
      )}
    </Group>
  )
}

/* ── Inline text staging panel (text tasks) ── */
function TextStagePanel({
  projectId,
  taskUsesClasses,
  classes,
  onStage,
}: {
  projectId: string
  taskUsesClasses: boolean
  classes: LabelClass[]
  onStage: (drafts: StagedDraft[]) => void
}) {
  const [split, setSplit] = useState<StagedSplit>('auto')
  const [classId, setClassId] = useState<string | null>(null)
  const [text, setText] = useState('')

  const handleAdd = () => {
    const trimmed = text.trim()
    if (!trimmed) return
    onStage([
      {
        kind: 'text',
        name: trimmed.length > 80 ? `${trimmed.slice(0, 80)}…` : trimmed,
        detail: `${trimmed.length} characters`,
        split,
        classId,
        text: trimmed,
      },
    ])
    setText('')
  }

  return (
    <Stack gap="sm">
      <StagingDefaults
        split={split}
        onSplitChange={setSplit}
        classId={classId}
        onClassChange={setClassId}
        classes={classes}
      />
      <Tabs defaultValue="type" keepMounted={false}>
        <Tabs.List mb="sm">
          <Tabs.Tab value="type">Type or paste</Tabs.Tab>
          <Tabs.Tab value="files">Import files</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="type">
          <Stack gap="sm">
            <Textarea
              label="Text"
              placeholder="Paste or type the text content for this item"
              autosize
              minRows={4}
              value={text}
              onChange={(e) => setText(e.currentTarget.value)}
            />
            <Group justify="flex-end">
              <Button onClick={handleAdd} disabled={!text.trim()}>
                Add item
              </Button>
            </Group>
          </Stack>
        </Tabs.Panel>
        <Tabs.Panel value="files">
          <TextBatchImporter
            projectId={projectId}
            taskUsesClasses={taskUsesClasses}
            split={split}
            classId={classId}
            classes={classes}
            onStage={onStage}
          />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}

/* ── The panel ── */
export function AddDataPanel({ projectId }: { projectId: string }) {
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(activeProject.task)
  // Boxes carry a class each, so there is no folder-per-class upload for them: files go up unlabelled, or with their
  // boxes through the YOLO import.
  const labelsWithBoxes = descriptor.annotation.type === 'bounding_box'
  const taskUsesClasses = descriptor.annotation.requiresLabelClasses && !labelsWithBoxes
  const { data: classesData } = useLabelClasses(taskUsesClasses ? projectId : undefined)
  // Tasks that don't label with classes (regression, captioning, ASR, generation) get no class controls at
  // all, whatever classes the project happens to hold, so no class can be attached to their uploads.
  const classes = useMemo(
    () => (taskUsesClasses ? (classesData?.classes ?? []) : []),
    [taskUsesClasses, classesData?.classes],
  )
  const classRefs = useMemo(() => classes.map((c) => ({ classId: c.classId, name: c.name })), [classes])
  const classNames = useMemo(() => new Map(classes.map((c) => [c.classId, c.name])), [classes])

  const isFileTask = descriptor.itemSpec.payload === 'file'
  // Files that go up with no class are still "done" unless the task labels them some other way afterwards.
  const labelledWithoutClass = !labelsWithBoxes && !(isFileTask && descriptor.annotation.type === 'text_sequence')
  const [settings, setSettings] = useState<CommitSettings>({ ratios: DEFAULT_RATIOS, stratify: true })
  const [yoloOpened, { open: openYolo, close: closeYolo }] = useDisclosure(false)

  const queue = useUploadQueue()
  const staging = useFileStaging({
    projectId,
    accept: descriptor.itemSpec.accept,
    taskUsesClasses,
    classes: classRefs,
  })
  const fileCommit = useFileCommit({ staging, classNames, projectId, taskUsesClasses, labelledWithoutClass, settings })
  const itemCommit = useItemCommit({
    items: queue.items,
    clear: queue.clear,
    classNames,
    projectId,
    taskUsesClasses,
    labelledWithoutClass,
    settings,
  })

  const fileSummary = useMemo<CommitSummary>(() => {
    const { totals } = staging.index
    return {
      total: totals.files,
      uploadable: totals.uploadable,
      bySplit: totals.bySplit,
      errors: totals.withErrors,
      warnings: totals.withWarnings,
      unclassified: taskUsesClasses ? totals.unclassified : 0,
      skipped: totals.skipped,
      newClasses: totals.newClassNames.length,
    }
  }, [staging.index, taskUsesClasses])

  const itemRows = useMemo(() => queue.items.map((item) => itemToRow(item, classes.length > 0)), [queue.items, classes])
  const itemSummary = useMemo<CommitSummary>(() => {
    const bySplit = { train: 0, validation: 0, test: 0, auto: 0 }
    let unclassified = 0
    for (const row of itemRows) {
      bySplit[row.split]++
      if (row.unclassified) unclassified++
    }
    return {
      total: itemRows.length,
      uploadable: itemRows.length,
      bySplit,
      errors: 0,
      warnings: 0,
      unclassified,
      skipped: 0,
      newClasses: 0,
    }
  }, [itemRows])
  const classOptions = useMemo(() => classes.map((c) => ({ value: existingKey(c.classId), label: c.name })), [classes])

  const editItems = (ids: string[], edit: RowEdit) => {
    const patch: StagedEdit = {}
    if (edit.split !== undefined) patch.split = edit.split
    if (edit.classKey !== undefined) patch.classId = classIdOfKey(edit.classKey)
    queue.edit(ids, patch)
  }

  const hint = labelsWithBoxes
    ? 'Drop images into split folders and upload them, then draw the boxes in Browse. Or import a dataset that is already labelled in YOLO format.'
    : isFileTask
      ? 'Drop files, folders or archives into class and split folders, check what was understood, then upload. Uploads carry on in the background.'
      : 'Stage items, review their split and class in the table, then upload. Uploads carry on in the background.'

  const staged = isFileTask ? fileSummary.total : itemSummary.total

  return (
    <div className="flex flex-col gap-3" style={{ height: '100%', minHeight: 0, overflow: 'hidden' }}>
      {/* The hint is for the empty drawer; once something is staged the room is better spent on the list. */}
      {((staged === 0 && !isFileTask) || labelsWithBoxes) && (
        <Group justify="space-between" wrap="nowrap" align="flex-start">
          <Text size="xs" c="dimmed">
            {hint}
          </Text>
          {labelsWithBoxes && (
            <Button size="compact-md" variant="default" leftSection={<CloudArrowUpIcon size={14} />} onClick={openYolo}>
              Import YOLO dataset
            </Button>
          )}
        </Group>
      )}

      <Modal opened={yoloOpened} onClose={closeYolo} title="Import a YOLO dataset" size="lg" centered>
        <YoloImporter projectId={projectId} />
      </Modal>

      {isFileTask ? (
        <div className="flex flex-col gap-3" style={{ flex: 1, minHeight: 0 }}>
          <FileImport
            staging={staging}
            classes={classes}
            taskUsesClasses={taskUsesClasses}
            accept={descriptor.itemSpec.accept}
            settings={settings}
            disabled={fileCommit.starting}
          />
          <CommitBar
            summary={fileSummary}
            settings={settings}
            onSettingsChange={setSettings}
            canStratify={taskUsesClasses}
            checking={staging.checking}
            starting={fileCommit.starting}
            locked={staging.busy !== null}
            onUpload={() => void fileCommit.start()}
            onDiscard={staging.clear}
            showProblems={false}
          />
        </div>
      ) : (
        // The row track must be capped (minmax(0,1fr)), or thousands of staged rows stretch it past the viewport.
        <div className="grid gap-3 md:grid-cols-[26rem_minmax(0,1fr)] grid-rows-[minmax(0,1fr)] overflow-hidden flex-1 min-h-0">
          <Paper p="md" style={{ overflowY: 'auto', minHeight: 0 }}>
            <SectionLabel mb="sm">Add items</SectionLabel>
            {descriptor.itemSpec.payload === 'inline_text' && (
              <TextStagePanel
                projectId={projectId}
                taskUsesClasses={taskUsesClasses}
                classes={classes}
                onStage={queue.stage}
              />
            )}
            {descriptor.itemSpec.payload === 'record' && (
              <TabularCsvImporter requiresLabelClasses={taskUsesClasses} classes={classes} onStage={queue.stage} />
            )}
          </Paper>

          <Stack gap="md" style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <ReviewTable
              rows={itemRows}
              classOptions={classOptions}
              showClass={classes.length > 0}
              onEdit={editItems}
              onRemove={queue.remove}
              emptyMessage={
                <Text size="sm" c="dimmed" ta="center" maw={360}>
                  Nothing staged yet — add items on the left, then review each one's split and class here before
                  uploading.
                </Text>
              }
            />
            <CommitBar
              summary={itemSummary}
              settings={settings}
              onSettingsChange={setSettings}
              canStratify={taskUsesClasses}
              starting={false}
              onUpload={itemCommit.start}
              onDiscard={queue.clear}
            />
          </Stack>
        </div>
      )}
    </div>
  )
}

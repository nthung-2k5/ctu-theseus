/**
 * Upload page – gets raw data into the project.
 *
 * File tasks (vision/audio) get a filesystem view: class folders holding
 * train/validation/test folders, filled by dropping files, folders or archives
 * (see FileTreeView). Text tasks type items inline and tabular tasks import a
 * CSV; for those the left column *stages* items and the right column previews
 * the batch, with each item's split and class editable, and owns the Upload
 * button that actually sends it.
 *
 * Either way, nothing is sent until the user presses Upload, and uploaded
 * items land in the mutable draft ready for snapshotting on the Dataset page.
 * This page doesn't browse the pool itself — see DatasetPage.
 */

import { Button, Group, Paper, Select, Stack, Textarea } from '@mantine/core'
import { FileTreeView } from '@public/components/dataset/FileTreeView'
import { FileUploadBar } from '@public/components/dataset/FileUploadBar'
import { TabularCsvImporter } from '@public/components/dataset/TabularCsvImporter'
import { UploadQueuePanel } from '@public/components/dataset/UploadQueuePanel'
import { PageHeader, SectionLabel } from '@public/components/ui'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import { projectDetailQueryOptions, useLabelClasses } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useFileStaging } from '@public/lib/upload/useFileStaging'
import { type StagedDraft, useUploadQueue } from '@public/lib/uploadQueue'
import type { LabelClass, SplitType } from '@public/store/types'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useMemo, useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/upload')

/**
 * The text entry panel picks a split/class up front purely as the *default* for
 * whatever it stages next — the preview pane is where these get corrected,
 * so nothing here is a final assignment.
 */
function StagingDefaults({
  split,
  onSplitChange,
  classId,
  onClassChange,
  classes,
}: {
  split: SplitType
  onSplitChange: (split: SplitType) => void
  classId: string | null
  onClassChange: (classId: string | null) => void
  classes: LabelClass[]
}) {
  return (
    <Group grow align="flex-start">
      <Select
        label="Split"
        data={SPLIT_OPTIONS}
        value={split}
        onChange={(v) => onSplitChange((v ?? 'train') as SplitType)}
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
function TextStagePanel({ classes, onStage }: { classes: LabelClass[]; onStage: (drafts: StagedDraft[]) => void }) {
  const [split, setSplit] = useState<SplitType>('train')
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
  )
}

/* ── Main Upload page ── */
export function UploadPage() {
  const { projectId } = routeApi.useParams()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(activeProject.task)
  const taskUsesClasses = descriptor.annotation.requiresLabelClasses
  const { data: classesData } = useLabelClasses(taskUsesClasses ? projectId : undefined)
  // Tasks that don't label with classes (regression, captioning, ASR, generation) get no class controls at
  // all, whatever classes the project happens to hold, so no class can be attached to their uploads.
  const classes = useMemo(
    () => (taskUsesClasses ? (classesData?.classes ?? []) : []),
    [taskUsesClasses, classesData?.classes],
  )
  const classRefs = useMemo(() => classes.map((c) => ({ classId: c.classId, name: c.name })), [classes])

  const isFileTask = descriptor.itemSpec.payload === 'file'
  const queue = useUploadQueue()
  const staging = useFileStaging({ accept: descriptor.itemSpec.accept, taskUsesClasses, classes: classRefs })
  const [uploadingFiles, setUploadingFiles] = useState(false)

  return (
    <div
      className="flex flex-col gap-3 p-3"
      style={{ height: 'calc(100vh - var(--app-shell-header-offset, 0rem))', overflow: 'hidden' }}
    >
      <PageHeader
        title="Upload"
        description={
          isFileTask
            ? 'Drop files, folders or archives into class and split folders, review them, then upload. Manage splits, classes and snapshots from the Dataset page.'
            : 'Stage items, review their split and class in the preview, then upload. Manage splits, classes and snapshots from the Dataset page.'
        }
      />

      {isFileTask ? (
        <div className="flex flex-col gap-3" style={{ flex: 1, minHeight: 0 }}>
          <FileUploadBar
            projectId={projectId}
            staging={staging}
            taskUsesClasses={taskUsesClasses}
            onUploadingChange={setUploadingFiles}
          />
          <Paper p="md" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <FileTreeView
              staging={staging}
              classes={classes}
              taskUsesClasses={taskUsesClasses}
              accept={descriptor.itemSpec.accept}
              disabled={uploadingFiles}
            />
          </Paper>
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-[30rem_1fr] overflow-hidden flex-1">
          <Paper p="md">
            <SectionLabel mb="sm">Add items</SectionLabel>
            {descriptor.itemSpec.payload === 'inline_text' && (
              <TextStagePanel classes={classes} onStage={queue.stage} />
            )}
            {descriptor.itemSpec.payload === 'record' && (
              <TabularCsvImporter requiresLabelClasses={taskUsesClasses} classes={classes} onStage={queue.stage} />
            )}
          </Paper>

          <UploadQueuePanel
            projectId={projectId}
            items={queue.items}
            classes={classes}
            onEdit={queue.edit}
            onEditAll={queue.editAll}
            onRemove={queue.remove}
            onClear={queue.clear}
          />
        </div>
      )}
    </div>
  )
}

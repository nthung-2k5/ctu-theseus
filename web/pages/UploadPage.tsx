/**
 * Upload page – gets raw data into the project: file drop for vision/audio
 * tasks, inline text entry for text tasks, CSV import for tabular tasks.
 *
 * The left column only *stages* items; the right column previews the whole
 * batch, with each item's split and class editable, and owns the Upload
 * button that actually sends it. Once uploaded, items land in the mutable
 * draft ready for snapshotting on the Dataset page. This page doesn't browse
 * the pool itself — see DatasetPage.
 */

import { Box, Button, Card, Group, Select, Stack, Text, Textarea, ThemeIcon, Title } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { CloudArrowUpIcon, UploadSimpleIcon } from '@phosphor-icons/react'
import { TabularCsvImporter } from '@public/components/dataset/TabularCsvImporter'
import { UploadQueuePanel } from '@public/components/dataset/UploadQueuePanel'
import { PageHeader } from '@public/components/ui'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import { formatBytes } from '@public/lib/format'
import { projectDetailQueryOptions, useLabelClasses } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { type StagedDraft, useUploadQueue } from '@public/lib/uploadQueue'
import type { LabelClass, SplitType } from '@public/store/types'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/upload')

/**
 * Both entry panels pick a split/class up front purely as the *default* for
 * whatever they stage next — the preview pane is where these get corrected,
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

/* ── File staging panel (vision/audio tasks) ── */
function FileStagePanel({
  accept,
  classes,
  onStage,
}: {
  accept?: string[]
  classes: LabelClass[]
  onStage: (drafts: StagedDraft[]) => void
}) {
  const [split, setSplit] = useState<SplitType>('train')
  const [classId, setClassId] = useState<string | null>(null)

  const handleDrop = (files: File[]) =>
    onStage(
      files.map((file) => ({
        kind: 'file' as const,
        name: file.name,
        detail: formatBytes(file.size),
        split,
        classId,
        file,
      })),
    )

  return (
    <Stack gap="sm" className="h-full">
      <StagingDefaults
        split={split}
        onSplitChange={setSplit}
        classId={classId}
        onClassChange={setClassId}
        classes={classes}
      />
      <Dropzone onDrop={handleDrop} accept={accept} radius="md" className="flex-1 grid place-items-center">
        <Group justify="center" gap="md" py="xl" style={{ pointerEvents: 'none' }}>
          <ThemeIcon size={44} variant="light" color="primary" radius="xl">
            <CloudArrowUpIcon size={24} />
          </ThemeIcon>
          <div>
            <Text fw={600}>Drop files here or click to browse</Text>
            <Text size="xs" c="dimmed">
              Nothing is uploaded until you review the batch and press Upload.
            </Text>
          </div>
        </Group>
      </Dropzone>
    </Stack>
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
  const { data: classesData } = useLabelClasses(descriptor.annotation.requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []

  const queue = useUploadQueue()

  return (
    <Box
      style={{
        height:
          'calc(100vh - (var(--app-shell-header-offset, 0rem) + var(--app-shell-padding) + var(--app-shell-footer-offset, 0rem) + var(--app-shell-padding)))',
      }}
    >
      <Stack gap="xl" className="h-full" style={{ overflow: 'hidden' }}>
        <PageHeader
          title="Upload"
          description="Stage items, review their split and class in the preview, then upload. Manage splits, classes, and snapshots from the Dataset page."
        />

        <div className="grid gap-6 md:grid-cols-[30rem_1fr] overflow-hidden flex-1">
          <Card withBorder p="lg" radius="md">
            <Group gap="sm" mb="md">
              <ThemeIcon size="md" variant="light" color="primary">
                <UploadSimpleIcon size={18} />
              </ThemeIcon>
              <Title order={5}>Add items</Title>
            </Group>
            {descriptor.itemSpec.payload === 'file' && (
              <FileStagePanel accept={descriptor.itemSpec.accept} classes={classes} onStage={queue.stage} />
            )}
            {descriptor.itemSpec.payload === 'inline_text' && (
              <TextStagePanel classes={classes} onStage={queue.stage} />
            )}
            {descriptor.itemSpec.payload === 'record' && (
              <TabularCsvImporter
                requiresLabelClasses={descriptor.annotation.requiresLabelClasses}
                classes={classes}
                onStage={queue.stage}
              />
            )}
          </Card>

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
      </Stack>
    </Box>
  )
}

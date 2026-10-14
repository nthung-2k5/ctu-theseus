/**
 * The file import: how vision / audio / video data gets into the project.
 *
 * Empty, it is one big drop target. Once something is staged it becomes a summary card (what was understood, in one
 * glance) over a body that can be read three ways: class lanes of previews (spot-check by eye), the folders as they
 * will be created, or a flat table (sort things out in bulk). All three edit the same staged files, and dropping
 * files anywhere — or onto a particular lane or folder — adds more.
 *
 * Nothing is sent from here: the footer's Upload button (see `CommitBar`) does that, and hands the batch to a
 * background job.
 */

import {
  Alert,
  Badge,
  Button,
  Group,
  Loader,
  Popover,
  Progress,
  ScrollArea,
  SegmentedControl,
  Stack,
  Text,
} from '@mantine/core'
import { CloudArrowUpIcon, FolderSimplePlusIcon, UploadSimpleIcon } from '@phosphor-icons/react'
import type { CommitSettings } from '@public/lib/upload/commit'
import { fileToRow } from '@public/lib/upload/review'
import type { DropContext } from '@public/lib/upload/types'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import type { LabelClass } from '@public/store/types'
import { type ChangeEvent, type DragEvent, useMemo, useRef, useState } from 'react'
import { ClassLanes } from './ClassLanes'
import { FolderTree } from './FolderTree'
import { ImportSummary } from './ImportSummary'
import { ReviewTable } from './ReviewTable'

/** Extensions that always go through the file picker, whatever the task accepts. */
const ARCHIVE_ACCEPT = ['.zip', '.tar', '.tar.gz', '.tgz', '.gz']

const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer.types).includes('Files')

type View = 'lanes' | 'folders' | 'table'

export function FileImport({
  staging,
  classes,
  taskUsesClasses,
  accept,
  settings,
  disabled,
}: {
  staging: FileStaging
  classes: LabelClass[]
  taskUsesClasses: boolean
  accept?: string[]
  settings: CommitSettings
  /** The batch is being handed to an upload: nothing may change under it. */
  disabled: boolean
}) {
  const { state, index, busy, stageFiles, stageDrop, cancel, move, remove } = staging
  const locked = disabled || busy !== null
  const empty = index.totals.files === 0

  const [view, setView] = useState<View>('lanes')
  const [dropKey, setDropKey] = useState<string | null>(null)
  const filesInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)
  const pickerDrop = useRef<DropContext>({})

  const openPicker = (kind: 'files' | 'folder', drop: DropContext) => {
    pickerDrop.current = drop
    ;(kind === 'files' ? filesInput : folderInput).current?.click()
  }

  const onPicked = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? [])
    event.currentTarget.value = '' // lets the same folder be picked again
    void stageFiles(files, pickerDrop.current)
  }

  const dropHandlers = (key: string, drop: DropContext) => ({
    onDragOver: (event: DragEvent) => {
      if (locked || !hasFiles(event)) return
      event.preventDefault()
      event.stopPropagation()
      event.dataTransfer.dropEffect = 'copy'
      setDropKey(key)
    },
    onDrop: (event: DragEvent) => {
      if (locked || !hasFiles(event)) return
      event.preventDefault()
      event.stopPropagation()
      setDropKey(null)
      void stageDrop(event.dataTransfer, drop)
    },
  })
  const rootDrop = dropHandlers('root', {})

  const classOptions = useMemo(
    () =>
      index.folders
        .filter((f) => !f.isNone)
        .map((f) => ({ value: f.key, label: f.pending ? `${f.name} (new class)` : f.name })),
    [index.folders],
  )
  const tableRows = useMemo(
    () => (view === 'table' ? [...state.files.values()].map((f) => fileToRow(f, taskUsesClasses)) : []),
    [view, state.files, taskUsesClasses],
  )
  const acceptAttr = [...(accept ?? []), ...ARCHIVE_ACCEPT].join(',')

  const dropZone = {
    onDragOver: rootDrop.onDragOver,
    onDrop: rootDrop.onDrop,
    onDragLeave: (event: DragEvent) => {
      if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropKey(null)
    },
  }
  const rootActive = dropKey === 'root'

  return (
    <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
      {busy && (
        <Alert variant="light" p="xs">
          <Group gap="sm" wrap="nowrap">
            <Loader size="xs" />
            <Text size="sm" style={{ flex: 1 }} truncate>
              {busy.label}…
            </Text>
            <Progress value={busy.fraction * 100} w={160} size="sm" />
            <Button size="compact-xs" variant="subtle" color="red" onClick={cancel}>
              Cancel
            </Button>
          </Group>
        </Alert>
      )}

      {empty ? (
        <section
          aria-label="Drop files here"
          {...dropZone}
          style={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            padding: 24,
            textAlign: 'center',
            border: `2px dashed ${rootActive ? 'var(--mantine-primary-color-filled)' : 'var(--mantine-color-default-border)'}`,
            borderRadius: 'var(--mantine-radius-md)',
            background: rootActive ? 'var(--mantine-primary-color-light)' : undefined,
          }}
        >
          <CloudArrowUpIcon size={44} />
          <Text fw={700} size="lg">
            Drop a folder, files or an archive here
          </Text>
          <Text size="sm" c="dimmed" maw={480}>
            {taskUsesClasses
              ? 'One folder per class (cat/, dog/…) is enough. Add train, validation and test folders inside them and those splits are kept; otherwise the files are split at random when you upload. '
              : 'Train, validation and test folders are kept as splits; otherwise the files are split at random when you upload. '}
            .zip, .tar and .tar.gz are unpacked here in your browser. Nothing is uploaded until you press Upload.
          </Text>
          <Group gap="xs">
            <Button
              leftSection={<FolderSimplePlusIcon size={16} />}
              disabled={locked}
              onClick={() => openPicker('folder', {})}
            >
              Choose a folder
            </Button>
            <Button
              variant="default"
              leftSection={<UploadSimpleIcon size={16} />}
              disabled={locked}
              onClick={() => openPicker('files', {})}
            >
              Choose files or archives
            </Button>
          </Group>
          {taskUsesClasses && classes.length > 0 && (
            <Stack gap={6} align="center">
              <Text size="xs" c="dimmed">
                Or drop straight into a class:
              </Text>
              <Group gap={6} justify="center">
                {classes.map((c) => {
                  const key = `c:${c.classId}`
                  const active = dropKey === key
                  return (
                    <Badge
                      key={c.classId}
                      component="button"
                      type="button"
                      size="lg"
                      variant={active ? 'filled' : 'light'}
                      disabled={locked}
                      style={{ cursor: locked ? 'default' : 'pointer' }}
                      onClick={() => openPicker('files', { classKey: key })}
                      {...dropHandlers(key, { classKey: key })}
                    >
                      {c.name}
                    </Badge>
                  )
                })}
              </Group>
            </Stack>
          )}
        </section>
      ) : (
        <>
          <ImportSummary staging={staging} taskUsesClasses={taskUsesClasses} settings={settings} locked={locked} />

          <Group justify="space-between" wrap="nowrap" align="flex-end">
            <Group gap="xs">
              <Button
                size="xs"
                variant="default"
                leftSection={<FolderSimplePlusIcon size={14} />}
                disabled={locked}
                onClick={() => openPicker('folder', {})}
              >
                Add folder
              </Button>
              <Button
                size="xs"
                variant="default"
                leftSection={<UploadSimpleIcon size={14} />}
                disabled={locked}
                onClick={() => openPicker('files', {})}
              >
                Add files or archives
              </Button>
              <Text size="xs" c="dimmed" visibleFrom="md">
                or drop more anywhere below
              </Text>
            </Group>
            <SegmentedControl
              size="xs"
              aria-label="How to show the staged files"
              data={[
                { value: 'lanes', label: 'Spot-check' },
                { value: 'folders', label: 'Folders' },
                { value: 'table', label: 'Table' },
              ]}
              value={view}
              onChange={(v) => setView(v as View)}
            />
          </Group>

          <section
            aria-label="Staged files — drop more files here to add them"
            {...dropZone}
            style={{
              flex: 1,
              minHeight: 0,
              display: 'flex',
              flexDirection: 'column',
              padding: view === 'folders' ? 0 : 8,
              border: `1px ${rootActive ? 'dashed' : 'solid'} ${
                rootActive ? 'var(--mantine-primary-color-filled)' : 'var(--mantine-color-default-border)'
              }`,
              borderRadius: 'var(--mantine-radius-md)',
              background: rootActive ? 'var(--mantine-primary-color-light)' : undefined,
            }}
          >
            {view === 'lanes' && (
              <ClassLanes
                staging={staging}
                classes={classes}
                taskUsesClasses={taskUsesClasses}
                locked={locked}
                openPicker={openPicker}
                onDropFiles={(dataTransfer, drop) => void stageDrop(dataTransfer, drop)}
              />
            )}
            {view === 'folders' && (
              <FolderTree
                staging={staging}
                classes={classes}
                taskUsesClasses={taskUsesClasses}
                locked={locked}
                openPicker={openPicker}
                dropHandlers={dropHandlers}
                dropKey={dropKey}
              />
            )}
            {view === 'table' && (
              <ReviewTable
                rows={tableRows}
                classOptions={classOptions}
                showClass={taskUsesClasses}
                onEdit={(ids, edit) => move(ids, edit)}
                onRemove={remove}
                disabled={locked}
                emptyMessage={null}
              />
            )}
          </section>
        </>
      )}

      {(state.junk > 0 || state.skipped.length > 0) && (
        <Group gap="md">
          {state.junk > 0 && (
            <Text size="xs" c="dimmed">
              {state.junk.toLocaleString()} system file{state.junk === 1 ? '' : 's'} ignored (.DS_Store, __MACOSX…)
            </Text>
          )}
          {state.skipped.length > 0 && (
            <Popover width={420} position="top-start" withArrow shadow="md">
              <Popover.Target>
                <Button size="compact-xs" variant="subtle" color="yellow">
                  {state.skipped.length.toLocaleString()} skipped — details
                </Button>
              </Popover.Target>
              <Popover.Dropdown>
                <ScrollArea.Autosize mah={240}>
                  <Stack gap={6}>
                    {state.skipped.slice(0, 100).map((s, i) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: a static, append-only list of paths that may repeat
                      <div key={i}>
                        <Text size="xs" fw={500} style={{ wordBreak: 'break-all' }}>
                          {s.path}
                        </Text>
                        <Text size="xs" c="dimmed">
                          {s.reason}
                        </Text>
                      </div>
                    ))}
                    {state.skipped.length > 100 && (
                      <Text size="xs" c="dimmed">
                        …and {state.skipped.length - 100} more
                      </Text>
                    )}
                  </Stack>
                </ScrollArea.Autosize>
              </Popover.Dropdown>
            </Popover>
          )}
        </Group>
      )}

      <input ref={filesInput} type="file" multiple accept={acceptAttr} hidden onChange={onPicked} />
      <input
        ref={folderInput}
        type="file"
        multiple
        hidden
        onChange={onPicked}
        {...({ webkitdirectory: '' } as Record<string, string>)}
      />
    </Stack>
  )
}

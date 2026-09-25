/**
 * The Upload page's filesystem view for file-based tasks (vision / audio).
 *
 * Class folders are the parents and `train` / `validation` / `test` their children. Files, folders and
 * archives are added by dropping them anywhere in the tree (the layout is detected from their paths) or onto
 * a specific folder (which then fixes that level), or through the "Add" buttons and each folder's menu.
 *
 * Nothing is sent from here: the view only stages and edits. Rendering goes through a virtualizer over a
 * flattened row list, so the DOM cost follows the viewport, not the number of files.
 */

import {
  Alert,
  Badge,
  Box,
  Button,
  Group,
  Loader,
  Popover,
  Progress,
  ScrollArea,
  Select,
  Stack,
  Text,
} from '@mantine/core'
import { CloudArrowUpIcon, FolderSimplePlusIcon, TrashIcon, UploadSimpleIcon, XIcon } from '@phosphor-icons/react'
import { NO_CLASS_KEY, SPLITS } from '@public/lib/upload/fileTree'
import { LAYOUT_LABELS, LAYOUT_LEVELS } from '@public/lib/upload/layout'
import { filesUnder, flattenTree, type TreeRow } from '@public/lib/upload/rows'
import type { DropContext, LayoutKind } from '@public/lib/upload/types'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import type { LabelClass, SplitType } from '@public/store/types'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type ChangeEvent, type DragEvent, type MouseEvent, useMemo, useRef, useState } from 'react'
import { FilePreview } from './FilePreview'
import { ClassRow, FileRow, type FolderActions, ROW_HEIGHT, SplitRow } from './FileTreeRows'
import { openMapToClass, openRenameFolder } from './FolderModals'

/** Extensions that always go through the file picker, whatever the task accepts. */
const ARCHIVE_ACCEPT = ['.zip', '.tar', '.tar.gz', '.tgz', '.gz']

const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer.types).includes('Files')

function MoveControls({
  classOptions,
  taskUsesClasses,
  onMove,
}: {
  classOptions: { value: string; label: string }[]
  taskUsesClasses: boolean
  onMove: (target: { classKey?: string | null; split?: SplitType }) => void
}) {
  const [opened, setOpened] = useState(false)
  const [classKey, setClassKey] = useState<string | null>(null)
  const [split, setSplit] = useState<SplitType | null>(null)
  return (
    <Popover opened={opened} onChange={setOpened} position="bottom-start" withArrow shadow="md">
      <Popover.Target>
        <Button size="xs" variant="light" onClick={() => setOpened((o) => !o)}>
          Move to…
        </Button>
      </Popover.Target>
      <Popover.Dropdown>
        <Stack gap="xs" w={220}>
          {taskUsesClasses && (
            <Select
              size="xs"
              label="Class"
              placeholder="Keep current"
              data={[...classOptions, { value: NO_CLASS_KEY, label: 'No class' }]}
              value={classKey}
              onChange={setClassKey}
              searchable
              clearable
            />
          )}
          <Select
            size="xs"
            label="Split"
            placeholder="Keep current"
            data={SPLITS.map((s) => ({ value: s, label: s }))}
            value={split}
            onChange={(v) => setSplit(v as SplitType | null)}
            clearable
          />
          <Button
            size="xs"
            disabled={!classKey && !split}
            onClick={() => {
              onMove({
                classKey: classKey === null ? undefined : classKey === NO_CLASS_KEY ? null : classKey,
                split: split ?? undefined,
              })
              setOpened(false)
              setClassKey(null)
              setSplit(null)
            }}
          >
            Move
          </Button>
        </Stack>
      </Popover.Dropdown>
    </Popover>
  )
}

export function FileTreeView({
  staging,
  classes,
  taskUsesClasses,
  accept,
  disabled,
}: {
  staging: FileStaging
  classes: LabelClass[]
  taskUsesClasses: boolean
  accept?: string[]
  /** An upload is running: the tree is read-only until it finishes. */
  disabled: boolean
}) {
  // TanStack Virtual returns functions the React Compiler would otherwise memoize away. That also switches off
  // its automatic memoization here, so everything that scales with the file count is memoized by hand below:
  // this component re-renders on every scroll frame.
  'use no memo'

  const { state, index, busy, stageFiles, stageDrop, relayout, renameFolder, cancel, move, remove, mapFolder } = staging
  const locked = disabled || busy !== null

  const [overrides, setOverrides] = useState<Record<string, boolean>>({})
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<number | null>(null)
  const [dropKey, setDropKey] = useState<string | null>(null)

  const parentRef = useRef<HTMLDivElement>(null)
  const filesInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)
  const pickerDrop = useRef<DropContext>({})

  const rows = useMemo(
    () => flattenTree(index.folders, overrides, taskUsesClasses),
    [index.folders, overrides, taskUsesClasses],
  )
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
    getItemKey: (i) => rows[i].key,
  })

  // Files that were uploaded (or removed) since they were selected drop out of the selection.
  const selectedIds = useMemo(() => [...selected].filter((id) => state.files.has(id)), [selected, state.files])
  const previewFile = selectedIds.length === 1 ? state.files.get(selectedIds[0]) : undefined

  const existingOptions = useMemo(() => classes.map((c) => ({ value: `c:${c.classId}`, label: c.name })), [classes])
  const classOptions = useMemo(
    () =>
      index.folders
        .filter((f) => !f.isNone)
        .map((f) => ({ value: f.key, label: f.pending ? `${f.name} (new class)` : f.name })),
    [index.folders],
  )

  /* ── Adding files ── */

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

  /* ── Selection ── */

  const commitSelection = (ids: Iterable<string>) => setSelected(new Set(ids))

  const selectionOf = (row: TreeRow): 0 | 1 | 2 => {
    if (selected.size === 0) return 0
    const files = filesUnder(row)
    const count = files.reduce((n, f) => n + (selected.has(f.id) ? 1 : 0), 0)
    return count === 0 ? 0 : count === files.length ? 2 : 1
  }

  const toggleFolderSelection = (row: TreeRow) => {
    const ids = filesUnder(row).map((f) => f.id)
    const all = ids.every((id) => selected.has(id))
    const next = new Set(selected)
    for (const id of ids) all ? next.delete(id) : next.add(id)
    setSelected(next)
  }

  const clickFile = (rowIndex: number, id: string, event: MouseEvent) => {
    if (event.shiftKey && anchor !== null) {
      const [from, to] = anchor < rowIndex ? [anchor, rowIndex] : [rowIndex, anchor]
      const range = rows.slice(from, to + 1).flatMap((r) => (r.type === 'file' ? [r.file.id] : []))
      commitSelection(event.ctrlKey || event.metaKey ? [...selected, ...range] : range)
      return
    }
    setAnchor(rowIndex)
    if (event.ctrlKey || event.metaKey) toggleId(id)
    else commitSelection([id])
  }

  const toggleId = (id: string) => {
    const next = new Set(selected)
    if (!next.delete(id)) next.add(id)
    setSelected(next)
  }

  /* ── Folder actions ── */

  const folderActions = (row: Extract<TreeRow, { type: 'class' | 'split' }>): FolderActions => ({
    onToggle: () => setOverrides((o) => ({ ...o, [row.key]: !row.open })),
    onPick: openPicker,
    onRemoveFiles: () => remove(filesUnder(row).map((f) => f.id)),
    ...(row.type === 'class' && row.node.pending
      ? {
          onRename: () => openRenameFolder(row.node.name, (name) => renameFolder(row.node.key, name)),
          onMapToClass: () =>
            openMapToClass(row.node.name, existingOptions, (classKey) => mapFolder(row.node.key, classKey)),
        }
      : {}),
  })

  const renderRow = (row: TreeRow, rowIndex: number) => {
    switch (row.type) {
      case 'class': {
        const drop = { classKey: row.node.isNone ? null : row.node.key }
        return (
          <ClassRow
            row={row}
            actions={folderActions(row)}
            selection={selectionOf(row)}
            onSelect={() => toggleFolderSelection(row)}
            dropActive={dropKey === row.key}
            dragHandlers={dropHandlers(row.key, drop)}
            disabled={locked}
          />
        )
      }
      case 'split':
        return (
          <SplitRow
            row={row}
            actions={folderActions(row)}
            selection={selectionOf(row)}
            onSelect={() => toggleFolderSelection(row)}
            dropActive={dropKey === row.key}
            dragHandlers={dropHandlers(row.key, { classKey: row.node.isNone ? null : row.node.key, split: row.split })}
            disabled={locked}
          />
        )
      case 'file':
        return (
          <FileRow
            row={row}
            taskUsesClasses={taskUsesClasses}
            selected={selected.has(row.file.id)}
            onClick={(event) => clickFile(rowIndex, row.file.id, event)}
            onCheck={() => {
              setAnchor(rowIndex)
              toggleId(row.file.id)
            }}
            onRemove={() => remove([row.file.id])}
            disabled={locked}
          />
        )
    }
  }

  const batch = state.lastBatch
  const acceptAttr = [...(accept ?? []), ...ARCHIVE_ACCEPT].join(',')

  return (
    <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
      <Group justify="space-between" wrap="nowrap" align="flex-end">
        <Group gap="xs">
          <Button
            size="xs"
            variant="default"
            leftSection={<UploadSimpleIcon size={14} />}
            disabled={locked}
            onClick={() => openPicker('files', {})}
          >
            Add files or archives
          </Button>
          <Button
            size="xs"
            variant="default"
            leftSection={<FolderSimplePlusIcon size={14} />}
            disabled={locked}
            onClick={() => openPicker('folder', {})}
          >
            Add folder
          </Button>
          <Text size="xs" c="dimmed" visibleFrom="md">
            or drop them anywhere below — onto a folder to put them there
          </Text>
        </Group>
        {batch && batch.candidates.length > 1 && (
          <Group gap="xs" wrap="nowrap">
            <Text size="xs" c="dimmed">
              Last drop ({batch.count.toLocaleString()} files) read as
            </Text>
            <Select
              size="xs"
              w={180}
              data={batch.candidates.map((kind) => ({ value: kind, label: LAYOUT_LABELS[kind] }))}
              value={batch.kind}
              onChange={(kind) => kind && relayout(kind as LayoutKind)}
              allowDeselect={false}
              disabled={locked}
            />
          </Group>
        )}
      </Group>

      {batch && batch.drop.split === undefined && !LAYOUT_LEVELS[batch.kind].includes('split') && batch.count > 0 && (
        <Text size="xs" c="dimmed">
          No train / validation / test folders found in the last drop, so its files went to <b>train</b>. Move them from
          the toolbar, or drop them onto a split folder.
        </Text>
      )}

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

      {selectedIds.length > 0 && (
        <Group gap="xs" wrap="nowrap">
          <Badge variant="light">{selectedIds.length.toLocaleString()} selected</Badge>
          {!locked && (
            <>
              <MoveControls
                classOptions={classOptions}
                taskUsesClasses={taskUsesClasses}
                onMove={(target) => move(selectedIds, target)}
              />
              <Button
                size="xs"
                variant="light"
                color="red"
                leftSection={<TrashIcon size={14} />}
                onClick={() => {
                  remove(selectedIds)
                  setSelected(new Set())
                }}
              >
                Remove
              </Button>
            </>
          )}
          <Button
            size="xs"
            variant="subtle"
            color="gray"
            leftSection={<XIcon size={14} />}
            onClick={() => setSelected(new Set())}
          >
            Clear selection
          </Button>
        </Group>
      )}

      <Group align="stretch" wrap="nowrap" gap="md" style={{ flex: 1, minHeight: 0 }}>
        <div
          ref={parentRef}
          role="tree"
          onDragOver={rootDrop.onDragOver}
          onDrop={rootDrop.onDrop}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropKey(null)
          }}
          style={{
            flex: 1,
            minWidth: 0,
            overflow: 'auto',
            border: `1px ${dropKey === 'root' ? 'dashed' : 'solid'} ${
              dropKey === 'root' ? 'var(--mantine-primary-color-filled)' : 'var(--mantine-color-default-border)'
            }`,
            borderRadius: 'var(--mantine-radius-md)',
            background: dropKey === 'root' ? 'var(--mantine-primary-color-light)' : undefined,
          }}
        >
          {rows.length === 0 ? (
            <Stack align="center" justify="center" gap="xs" h="100%" p="xl" style={{ pointerEvents: 'none' }}>
              <CloudArrowUpIcon size={36} />
              <Text fw={600}>Drop files, folders or archives here</Text>
              <Text size="xs" c="dimmed" ta="center" maw={420}>
                Organize them as {taskUsesClasses ? 'class / train, validation, test' : 'train, validation, test'}{' '}
                folders and they are sorted for you. .zip, .tar and .tar.gz are unpacked here in your browser. Nothing
                is uploaded until you press Upload.
              </Text>
            </Stack>
          ) : (
            <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
              {virtualizer.getVirtualItems().map((item) => (
                <div
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
                  {renderRow(rows[item.index], item.index)}
                </div>
              ))}
            </div>
          )}
        </div>

        {previewFile && (
          <Box w={240} style={{ flexShrink: 0, overflow: 'hidden' }}>
            <ScrollArea h="100%">
              <FilePreview file={previewFile} taskUsesClasses={taskUsesClasses} />
            </ScrollArea>
          </Box>
        )}
      </Group>

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

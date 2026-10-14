/**
 * The folder view of the staged files: class folders, each holding `train` / `validation` / `test` folders (and an
 * `auto-split` bucket for files whose split is still to be drawn).
 *
 * It is the detail view for fixing things by folder. The shell around it (`FileImport`) owns adding files, the
 * pickers and the drop target, and hands in what the folder rows need to accept a drop of their own. Nothing is
 * sent from here. Rendering goes through a virtualizer over a flattened row list, so the DOM cost follows the
 * viewport, not the number of files.
 */

import { Badge, Box, Button, Group, Popover, ScrollArea, Select, Stack } from '@mantine/core'
import { TrashIcon, XIcon } from '@phosphor-icons/react'
import { NO_CLASS_KEY, STAGED_SPLITS } from '@public/lib/upload/fileTree'
import { filesUnder, flattenTree, type TreeRow } from '@public/lib/upload/rows'
import type { DropContext, StagedSplit } from '@public/lib/upload/types'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import type { LabelClass } from '@public/store/types'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type DragEvent, type KeyboardEvent, type MouseEvent, useMemo, useRef, useState } from 'react'
import { FilePreview } from './FilePreview'
import { ClassRow, FileRow, type FolderActions, ROW_HEIGHT, RowNavContext, SplitRow } from './FileTreeRows'
import { openMapToClass, openRenameFolder } from './FolderModals'

function MoveControls({
  classOptions,
  taskUsesClasses,
  onMove,
}: {
  classOptions: { value: string; label: string }[]
  taskUsesClasses: boolean
  onMove: (target: { classKey?: string | null; split?: StagedSplit }) => void
}) {
  const [opened, setOpened] = useState(false)
  const [classKey, setClassKey] = useState<string | null>(null)
  const [split, setSplit] = useState<StagedSplit | null>(null)
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
            data={STAGED_SPLITS.map((s) => ({ value: s, label: s === 'auto' ? 'auto-split' : s }))}
            value={split}
            onChange={(v) => setSplit(v as StagedSplit | null)}
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

export function FolderTree({
  staging,
  classes,
  taskUsesClasses,
  locked,
  openPicker,
  dropHandlers,
  dropKey,
}: {
  staging: FileStaging
  classes: LabelClass[]
  taskUsesClasses: boolean
  /** Something is being read or started, so the tree is read-only. */
  locked: boolean
  /** Opens the file or folder picker aimed at a class / split folder. */
  openPicker: (kind: 'files' | 'folder', drop: DropContext) => void
  /** Drag handlers that stage dropped files into the folder `drop` names (`key` highlights it while dragged over). */
  dropHandlers: (
    key: string,
    drop: DropContext,
  ) => { onDragOver: (e: DragEvent) => void; onDrop: (e: DragEvent) => void }
  dropKey: string | null
}) {
  // TanStack Virtual returns functions the React Compiler would otherwise memoize away. That also switches off
  // its automatic memoization here, so everything that scales with the file count is memoized by hand below:
  // this component re-renders on every scroll frame.
  'use no memo'

  const { state, index, renameFolder, move, remove, mapFolder } = staging

  const [overrides, setOverrides] = useState<Record<string, boolean>>({})
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<number | null>(null)
  const parentRef = useRef<HTMLDivElement>(null)

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

  // Keyboard: one row is tabbable at a time (a roving tabindex), and the arrow keys move it.
  const [activeRow, setActiveRow] = useState(0)
  const mountedRows = virtualizer.getVirtualItems()
  // If the active row has scrolled out of the virtualized window the tree would have no tab stop at all.
  const tabbableRow = mountedRows.some((v) => v.index === activeRow) ? activeRow : (mountedRows[0]?.index ?? 0)

  const focusRow = (target: number) => {
    const index = Math.max(0, Math.min(rows.length - 1, target))
    setActiveRow(index)
    virtualizer.scrollToIndex(index)
    // The row may not be mounted until after the scroll, so focus it on the next frame.
    requestAnimationFrame(() =>
      parentRef.current?.querySelector<HTMLElement>(`[role="treeitem"][data-row-index="${index}"]`)?.focus(),
    )
  }

  const onTreeKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const row = event.target as HTMLElement
    if (row.getAttribute('role') !== 'treeitem') return // a checkbox or menu inside a row keeps its own keys
    const index = Number(row.dataset.rowIndex)
    const expanded = row.getAttribute('aria-expanded')
    switch (event.key) {
      case 'ArrowDown':
        focusRow(index + 1)
        break
      case 'ArrowUp':
        focusRow(index - 1)
        break
      case 'Home':
        focusRow(0)
        break
      case 'End':
        focusRow(rows.length - 1)
        break
      case 'ArrowRight':
        if (expanded === 'false') row.click()
        break
      case 'ArrowLeft':
        if (expanded === 'true') row.click()
        break
      case 'Enter':
      case ' ':
        row.click() // opens or closes a folder, selects a file
        break
      default:
        return
    }
    event.preventDefault()
  }

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

  return (
    <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
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
          aria-label="Files to upload"
          onKeyDown={onTreeKeyDown}
          style={{
            flex: 1,
            minWidth: 0,
            overflow: 'auto',
            border: '1px solid var(--mantine-color-default-border)',
            borderRadius: 'var(--mantine-radius-md)',
          }}
        >
          {rows.length === 0 ? null : (
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
                  <RowNavContext.Provider
                    value={{
                      index: item.index,
                      tabbable: item.index === tabbableRow,
                      onFocus: () => setActiveRow(item.index),
                    }}
                  >
                    {renderRow(rows[item.index], item.index)}
                  </RowNavContext.Provider>
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
    </Stack>
  )
}

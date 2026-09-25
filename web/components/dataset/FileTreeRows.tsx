/**
 * The three row kinds of the Upload page's filesystem view: class folders, split folders and files.
 *
 * Rows are deliberately plain (text, icons, one checkbox) — no per-row `Select`s or thumbnails — because the
 * view renders through a virtualizer and every row that scrolls into view is mounted from scratch.
 */

import { ActionIcon, Badge, Checkbox, Group, Menu, Text, Tooltip } from '@mantine/core'
import {
  CaretDownIcon,
  CaretRightIcon,
  DotsThreeVerticalIcon,
  FileAudioIcon,
  FileIcon,
  FileImageIcon,
  FolderIcon,
  FolderOpenIcon,
  FolderSimplePlusIcon,
  LinkIcon,
  PencilSimpleIcon,
  TrashIcon,
  UploadSimpleIcon,
  WarningIcon,
  XIcon,
} from '@phosphor-icons/react'
import { formatBytes } from '@public/lib/format'
import { fileIssues, NO_CLASS_MESSAGE } from '@public/lib/upload/fileTree'
import type { TreeRow } from '@public/lib/upload/rows'
import type { DropContext } from '@public/lib/upload/types'
import type { CSSProperties, DragEvent, MouseEvent, ReactNode } from 'react'

export const ROW_HEIGHT = 32
const INDENT = 22

const SPLIT_LABEL = { train: 'train', validation: 'validation', test: 'test' } as const

/** What a folder row can do to its own contents. `onPick` opens the file or folder picker aimed at this folder. */
export interface FolderActions {
  onToggle: () => void
  onPick: (kind: 'files' | 'folder', drop: DropContext) => void
  onRemoveFiles: () => void
  onRename?: () => void
  onMapToClass?: () => void
}

interface RowFrame {
  depth: number
  selected?: boolean
  dropActive?: boolean
  children: ReactNode
  onClick?: (event: MouseEvent) => void
  dragHandlers?: {
    onDragOver: (event: DragEvent) => void
    onDrop: (event: DragEvent) => void
  }
}

function RowFrame({ depth, selected, dropActive, children, onClick, dragHandlers }: RowFrame) {
  const style: CSSProperties = {
    height: ROW_HEIGHT,
    paddingLeft: 8 + depth * INDENT,
    paddingRight: 8,
    background: dropActive
      ? 'var(--mantine-primary-color-light-hover)'
      : selected
        ? 'var(--mantine-primary-color-light)'
        : undefined,
    outline: dropActive ? '1px dashed var(--mantine-primary-color-filled)' : undefined,
    outlineOffset: -1,
  }
  return (
    <div
      role="treeitem"
      aria-selected={selected}
      tabIndex={-1}
      className="flex items-center gap-2 cursor-default select-none hover:bg-[var(--mantine-color-default-hover)]"
      style={style}
      onClick={onClick}
      onKeyDown={undefined}
      {...dragHandlers}
    >
      {children}
    </div>
  )
}

function FolderMenu({ actions, drop, pending }: { actions: FolderActions; drop: DropContext; pending?: boolean }) {
  return (
    <Menu position="bottom-end" withinPortal>
      <Menu.Target>
        <ActionIcon
          size="sm"
          variant="subtle"
          color="gray"
          aria-label="Folder actions"
          onClick={(e) => e.stopPropagation()}
        >
          <DotsThreeVerticalIcon size={16} />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown onClick={(e) => e.stopPropagation()}>
        <Menu.Item leftSection={<UploadSimpleIcon size={14} />} onClick={() => actions.onPick('files', drop)}>
          Add files or archives…
        </Menu.Item>
        <Menu.Item leftSection={<FolderSimplePlusIcon size={14} />} onClick={() => actions.onPick('folder', drop)}>
          Add folder…
        </Menu.Item>
        {pending && actions.onRename && (
          <Menu.Item leftSection={<PencilSimpleIcon size={14} />} onClick={actions.onRename}>
            Rename new class…
          </Menu.Item>
        )}
        {pending && actions.onMapToClass && (
          <Menu.Item leftSection={<LinkIcon size={14} />} onClick={actions.onMapToClass}>
            Use an existing class…
          </Menu.Item>
        )}
        <Menu.Divider />
        <Menu.Item color="red" leftSection={<TrashIcon size={14} />} onClick={actions.onRemoveFiles}>
          Remove all files here
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  )
}

interface SelectionProps {
  /** 0 = none, 1 = some, 2 = all of the files under this row. */
  selection: 0 | 1 | 2
  onSelect: () => void
  disabled?: boolean
}

function RowCheckbox({ selection, onSelect, disabled }: SelectionProps) {
  return (
    <Checkbox
      size="xs"
      aria-label="Select"
      checked={selection === 2}
      indeterminate={selection === 1}
      disabled={disabled}
      onChange={onSelect}
      onClick={(e) => e.stopPropagation()}
    />
  )
}

export function ClassRow({
  row,
  actions,
  selection,
  onSelect,
  dropActive,
  dragHandlers,
  disabled,
}: SelectionProps & {
  row: Extract<TreeRow, { type: 'class' }>
  actions: FolderActions
  dropActive: boolean
  dragHandlers: RowFrame['dragHandlers']
}) {
  const { node, open } = row
  return (
    <RowFrame depth={0} dropActive={dropActive} dragHandlers={dragHandlers} onClick={actions.onToggle}>
      {open ? <CaretDownIcon size={12} /> : <CaretRightIcon size={12} />}
      <RowCheckbox selection={selection} onSelect={onSelect} disabled={disabled} />
      {open ? <FolderOpenIcon size={18} weight="fill" /> : <FolderIcon size={18} weight="fill" />}
      <Text size="sm" fw={600} truncate style={{ minWidth: 0 }}>
        {node.name}
      </Text>
      <Group gap={6} wrap="nowrap" ml="auto">
        {node.pending && (
          <Tooltip label="No class with this name yet — it will be created when you upload">
            <Badge size="xs" variant="light" color="teal">
              new class
            </Badge>
          </Tooltip>
        )}
        {node.isNone && (
          <Badge size="xs" variant="light" color="yellow">
            unassigned
          </Badge>
        )}
        <Text size="xs" c="dimmed">
          {node.total} file{node.total === 1 ? '' : 's'}
        </Text>
        {!disabled && !node.isNone && (
          <FolderMenu actions={actions} drop={{ classKey: node.key }} pending={node.pending} />
        )}
      </Group>
    </RowFrame>
  )
}

export function SplitRow({
  row,
  actions,
  selection,
  onSelect,
  dropActive,
  dragHandlers,
  disabled,
}: SelectionProps & {
  row: Extract<TreeRow, { type: 'split' }>
  actions: FolderActions
  dropActive: boolean
  dragHandlers: RowFrame['dragHandlers']
}) {
  const { node, split, count, open, depth } = row
  return (
    <RowFrame depth={depth} dropActive={dropActive} dragHandlers={dragHandlers} onClick={actions.onToggle}>
      {open ? <CaretDownIcon size={12} /> : <CaretRightIcon size={12} />}
      <RowCheckbox selection={selection} onSelect={onSelect} disabled={disabled || count === 0} />
      {open ? <FolderOpenIcon size={18} /> : <FolderIcon size={18} />}
      <Text size="sm" truncate>
        {SPLIT_LABEL[split]}
      </Text>
      <Group gap={6} wrap="nowrap" ml="auto">
        <Text size="xs" c={count === 0 ? 'dimmed' : undefined}>
          {count}
        </Text>
        {!disabled && <FolderMenu actions={actions} drop={{ classKey: node.isNone ? null : node.key, split }} />}
      </Group>
    </RowFrame>
  )
}

function fileIcon(mime: string) {
  if (mime.startsWith('image/')) return <FileImageIcon size={16} />
  if (mime.startsWith('audio/')) return <FileAudioIcon size={16} />
  return <FileIcon size={16} />
}

export function FileRow({
  row,
  taskUsesClasses,
  selected,
  onClick,
  onCheck,
  onRemove,
  disabled,
}: {
  row: Extract<TreeRow, { type: 'file' }>
  taskUsesClasses: boolean
  selected: boolean
  onClick: (event: MouseEvent) => void
  onCheck: () => void
  onRemove: () => void
  disabled?: boolean
}) {
  const { file } = row
  // A "no class" file already sits in the "unassigned" folder, so don't repeat that warning on every row.
  const issues = fileIssues(file, taskUsesClasses).filter((i) => i.message !== NO_CLASS_MESSAGE)
  const hasError = issues.some((i) => i.severity === 'error')
  // Only a file that can never upload is struck through; one the server refused is still retryable.
  const excluded = file.issues.some((i) => i.severity === 'error')
  return (
    <RowFrame depth={row.depth} selected={selected} onClick={onClick}>
      <span style={{ width: 12 }} />
      <RowCheckbox selection={selected ? 2 : 0} onSelect={onCheck} disabled={disabled} />
      {fileIcon(file.file.type)}
      <Text
        size="sm"
        truncate
        style={{ minWidth: 0 }}
        td={excluded ? 'line-through' : undefined}
        c={excluded ? 'dimmed' : undefined}
      >
        {file.name}
      </Text>
      <Group gap={6} wrap="nowrap" ml="auto">
        {issues.length > 0 && (
          <Tooltip multiline maw={320} label={issues.map((issue) => <div key={issue.message}>{issue.message}</div>)}>
            <WarningIcon
              size={16}
              weight="fill"
              color={hasError ? 'var(--mantine-color-red-6)' : 'var(--mantine-color-yellow-6)'}
            />
          </Tooltip>
        )}
        <Text size="xs" c="dimmed">
          {formatBytes(file.size)}
        </Text>
        {!disabled && (
          <ActionIcon
            size="sm"
            variant="subtle"
            color="gray"
            aria-label={`Remove ${file.name}`}
            onClick={(e) => {
              e.stopPropagation()
              onRemove()
            }}
          >
            <XIcon size={12} />
          </ActionIcon>
        )}
      </Group>
    </RowFrame>
  )
}

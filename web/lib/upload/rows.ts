/**
 * Flattens the folder tree into the single list of rows the virtualized view renders.
 *
 * Only expanded folders contribute their children, so the row count tracks what the user has opened rather
 * than how many files are staged.
 */

import type { SplitType } from '../../store/types'
import { type FolderNode, SPLITS } from './fileTree'
import type { StagedFile } from './types'

/** Folders holding more files than this start collapsed. */
export const AUTO_OPEN_LIMIT = 200

export type TreeRow =
  | { type: 'class'; key: string; node: FolderNode; open: boolean }
  | { type: 'split'; key: string; node: FolderNode; split: SplitType; count: number; open: boolean; depth: number }
  | { type: 'file'; key: string; file: StagedFile; depth: number }

export const classRowKey = (nodeKey: string) => `class:${nodeKey}`
export const splitRowKey = (nodeKey: string, split: SplitType) => `split:${nodeKey}:${split}`

/** `overrides` holds what the user toggled by hand; anything absent falls back to the size-based default. */
export function flattenTree(
  folders: FolderNode[],
  overrides: Record<string, boolean>,
  taskUsesClasses: boolean,
): TreeRow[] {
  const rows: TreeRow[] = []
  for (const node of folders) {
    let splitDepth = 0
    if (taskUsesClasses) {
      const key = classRowKey(node.key)
      const open = overrides[key] ?? (node.total > 0 && node.total <= AUTO_OPEN_LIMIT)
      rows.push({ type: 'class', key, node, open })
      if (!open) continue
      splitDepth = 1
    }
    for (const split of SPLITS) {
      const files = node.files[split]
      const key = splitRowKey(node.key, split)
      const open = overrides[key] ?? (files.length > 0 && files.length <= AUTO_OPEN_LIMIT)
      rows.push({ type: 'split', key, node, split, count: files.length, open, depth: splitDepth })
      if (!open) continue
      for (const file of files) rows.push({ type: 'file', key: file.id, file, depth: splitDepth + 1 })
    }
  }
  return rows
}

/** Every staged file at or under a folder row. */
export function filesUnder(row: TreeRow): StagedFile[] {
  if (row.type === 'file') return [row.file]
  if (row.type === 'split') return row.node.files[row.split]
  return SPLITS.flatMap((split) => row.node.files[split])
}

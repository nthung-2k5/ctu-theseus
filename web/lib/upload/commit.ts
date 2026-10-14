/**
 * Turns what is staged into what a job sends: decides every split that was left to `auto`, drops what need not be
 * sent, and shapes each file or row into a unit. Pure, so the rules (which files stay behind, how the draw is
 * grouped) can be tested without a browser.
 */

import type { StagedItem } from '../uploadQueue'
import { hasBlockingError, isPendingKey } from './fileTree'
import { type Ratios, resolveSplits, type SplitPlan } from './splitPlan'
import type { PendingFolder, StagedFile } from './types'
import type { InlinePayload, InlineUnit } from './uploadInline'
import type { FileUnit } from './uploadStaged'

export interface CommitSettings {
  ratios: Ratios
  /** Keep every class in every split when drawing. Only meaningful for a task that labels with classes. */
  stratify: boolean
}

interface CommitOptions {
  settings: CommitSettings
  taskUsesClasses: boolean
  /** Whether an item with no class still counts as labelled (false where the label is drawn or typed afterwards). */
  labelledWithoutClass: boolean
  /** Injected for tests. */
  random?: () => number
}

export interface FileCommit {
  units: FileUnit[]
  /** Staged files that leave the staging area now: the ones being sent and the ones that need not be. */
  consumedIds: string[]
  /** How many of those need not be sent because the dataset already has their content. */
  alreadyThere: number
  /** What the draw of `auto` splits came to; `null` when nothing was left to draw. */
  drawn: SplitPlan | null
}

export function planFileCommit(files: Iterable<StagedFile>, options: CommitOptions): FileCommit {
  const { settings, taskUsesClasses, labelledWithoutClass } = options
  const sendable: StagedFile[] = []
  const consumedIds: string[] = []
  let alreadyThere = 0
  for (const file of files) {
    if (hasBlockingError(file)) continue // stays staged, for the user to fix or remove
    consumedIds.push(file.id)
    if (file.skip) alreadyThere++
    else sendable.push(file)
  }

  const { splits, drawn } = resolveSplits(sendable, {
    splitOf: (file) => file.split,
    groupOf: (file) => file.classKey,
    ratios: settings.ratios,
    stratify: settings.stratify && taskUsesClasses,
    random: options.random,
  })

  const units = sendable.map(
    (file): FileUnit => ({
      id: file.id,
      name: file.name,
      split: splits.get(file.id) ?? 'train',
      labelled: taskUsesClasses ? file.classKey !== null : labelledWithoutClass,
      weight: file.size,
      payload: { file: file.file, classKey: taskUsesClasses ? file.classKey : null },
    }),
  )
  return { units, consumedIds, alreadyThere, drawn }
}

/** The body of one staged item, as `POST /projects/:id/items` takes it (the split is the unit's). */
export function toInlinePayload(item: StagedItem): InlinePayload {
  const annotations =
    item.targetValue != null
      ? [{ annotationType: 'classification' as const, labelStructured: { value: item.targetValue } }]
      : item.classId
        ? [{ annotationType: 'classification' as const, classId: item.classId }]
        : undefined

  return {
    ...(item.text != null ? { textFeatures: { rawText: item.text } } : {}),
    ...(item.featuresJson ? { tabularFeatures: { featuresJson: item.featuresJson } } : {}),
    annotations,
  }
}

export interface ItemCommit {
  units: InlineUnit[]
  drawn: SplitPlan | null
}

export function planItemCommit(items: StagedItem[], options: CommitOptions): ItemCommit {
  const { settings, taskUsesClasses, labelledWithoutClass } = options
  const { splits, drawn } = resolveSplits(items, {
    splitOf: (item) => item.split,
    groupOf: (item) => item.classId,
    ratios: settings.ratios,
    stratify: settings.stratify && taskUsesClasses,
    random: options.random,
  })
  const units = items.map(
    (item): InlineUnit => ({
      id: item.id,
      name: item.name,
      split: splits.get(item.id) ?? 'train',
      labelled: item.targetValue != null || (taskUsesClasses ? item.classId !== null : labelledWithoutClass),
      weight: 1,
      payload: toInlinePayload(item),
    }),
  )
  return { units, drawn }
}

/** Names of the classes this batch will create: folders that no existing class matches and a file is still sent to. */
export function newClassNamesOf(units: FileUnit[], pending: ReadonlyMap<string, PendingFolder>): string[] {
  const keys = new Set<string>()
  for (const unit of units) if (isPendingKey(unit.payload.classKey)) keys.add(unit.payload.classKey)
  return [...keys].flatMap((key) => pending.get(key)?.name ?? [])
}

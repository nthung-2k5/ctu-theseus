import type { Annotation } from '@public/store/types'

/**
 * The item shape both the Dataset page's draft viewer and the Snapshot detail
 * page's per-version viewer render — a superset of every modality's fields,
 * since `GET /projects/{id}/items` always joins all four feature tables
 * regardless of the dataset's modality (see server/theseus/routers/datasets.py).
 */
export interface DatasetListItem {
  id: string
  externalId: string | null
  splitType: string
  createdAt: string | Date
  annotations?: Annotation[]
  downloadUrl?: string | null
  textFeatures?: { rawText: string } | null
  visionFeatures?: { width: number; height: number; imageFormat: string | null } | null
  audioFeatures?: { durationSeconds: string | number; sampleRateHz: number } | null
  tabularFeatures?: { featuresJson: unknown } | null
  /** Set on any derived item (preprocessed, augmented, or both): the pool original it came from. */
  sourceItemId?: string | null
  sourceExternalId?: string | null
  /** The preprocessing ops that produced this item; carried forward onto an augmented copy too. */
  preprocessing?: unknown
  /** Set only on an augmented copy. */
  augmentation?: unknown
}

/** Per-item checkbox selection, for pages with batch operations (e.g. the Dataset draft page). */
export interface ItemSelection {
  selectedIds: ReadonlySet<string>
  /** Toggle one item; `range` (shift-click) selects everything between the last toggled item and this one. */
  onToggle: (itemId: string, range?: boolean) => void
  /**
   * Set every id to `selected` in one update. The header checkbox needs this:
   * toggling each id individually *inverts* a mixed selection instead of
   * selecting all of it, which is the opposite of what an indeterminate
   * header checkbox means.
   */
  onToggleMany: (itemIds: string[], selected: boolean) => void
}

/**
 * Free-text ground truth (image/audio captioning, ASR) has no shared value to
 * bulk-assign across items the way a class does — every item needs its own
 * caption/transcript — so this is the one annotation type edited per-item,
 * from the detail modal, rather than through the bulk toolbar.
 */
export interface CaptionEditing {
  /** e.g. "Caption" or "Transcript" — used as the field label and placeholder. */
  label: string
  onSave: (itemId: string, existingAnnotationId: string | null, text: string) => Promise<void>
}

export const findClassAnnotation = (annotations: Annotation[] | undefined) => annotations?.find((a) => a.classId)

export const findTextSequenceAnnotation = (annotations: Annotation[] | undefined) =>
  annotations?.find((a) => a.annotationType === 'text_sequence')

export const tabularRow = (featuresJson: unknown): Record<string, unknown> =>
  featuresJson && typeof featuresJson === 'object' ? (featuresJson as Record<string, unknown>) : {}

import type { Annotation, LabelClass } from '@public/store/types'
import type { SPLIT_TYPES } from '../VersionBrowsing'

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
  /** A small JPEG of an image item, for grids; absent for anything that isn't an image. */
  thumbnailUrl?: string | null
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
  isSelected: (itemId: string) => boolean
  /** Toggle one item; `range` (shift-click) selects everything between the last toggled item and this one. */
  onToggle: (itemId: string, range?: boolean) => void
  /**
   * Set every id to `selected` in one update. The header checkbox needs this:
   * toggling each id individually *inverts* a mixed selection instead of
   * selecting all of it, which is the opposite of what an indeterminate
   * header checkbox means.
   */
  onToggleMany: (itemIds: string[], selected: boolean) => void
  /** Drop the whole selection (Escape in the grid). */
  onClear?: () => void
}

/**
 * Free-text ground truth (captioning, ASR, text generation) has no shared value
 * to bulk-assign across items the way a class does — every item needs its own
 * text — so it is edited per-item, from the detail drawer, rather than through
 * the bulk toolbar.
 */
export interface CaptionEditing {
  /** e.g. "Caption", "Transcript" or "Completion" — used as the field label and placeholder. */
  label: string
  onSave: (itemId: string, existingAnnotationId: string | null, text: string) => Promise<void>
}

/**
 * What the draft page lets you change about one item from its drawer and row menu. Omit it for a
 * read-only view (a snapshot is immutable), and the drawer and menu drop every control that would write.
 */
export interface ItemEditing {
  /** Empty unless the task labels with classes. */
  classes: LabelClass[]
  onSetSplit: (itemId: string, split: (typeof SPLIT_TYPES)[number]) => void
  onSetClass: (itemId: string, classId: string) => void
  /** A regression target: creates the annotation, or updates `existingAnnotationId` in place. */
  onSaveTarget: (itemId: string, existingAnnotationId: string | null, value: number) => Promise<void>
  onDelete: (itemId: string) => void
}

export const findTextSequenceAnnotation = (annotations: Annotation[] | undefined) =>
  annotations?.find((a) => a.annotationType === 'text_sequence')

export const tabularRow = (featuresJson: unknown): Record<string, unknown> =>
  featuresJson && typeof featuresJson === 'object' ? (featuresJson as Record<string, unknown>) : {}

import type { AnnotationType, DatasetModality, ProjectTask } from '@server/lib/enums'

/** Ludwig ECD/LLM feature type strings, as they appear in a compiled config. */
export type LudwigFeatureType = 'image' | 'text' | 'audio' | 'number' | 'category' | 'binary' | 'sequence' | 'vector'

/** Ludwig's built-in image augmentation ops (ludwig.schema.features.augmentation.image) — image input features only. */
export const ImageAugmentationTypes = [
  'random_horizontal_flip',
  'random_vertical_flip',
  'random_rotate',
  'random_blur',
  'random_brightness',
  'random_contrast',
] as const

export type ImageAugmentationType = (typeof ImageAugmentationTypes)[number]

/** A common, dependency-free subset of Ludwig's registered optimizers (ludwig.schema.optimizers) — enough variety to matter, none needing bitsandbytes. */
export const LudwigOptimizerTypes = ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'] as const

export type LudwigOptimizerType = (typeof LudwigOptimizerTypes)[number]

export interface LudwigInputFeature {
  name: string
  type: LudwigFeatureType
  column: string
  encoder?: { type: string; [key: string]: unknown }
  augmentation?: { type: ImageAugmentationType }[]
  /** Per-feature preprocessing overrides — currently only used to resize an image input (`height`/`width`). */
  preprocessing?: { height?: number; width?: number; [key: string]: unknown }
  [key: string]: unknown
}

export interface LudwigOutputFeature {
  name: string
  type: LudwigFeatureType
  column: string
  loss?: { type: string; [key: string]: unknown }
  [key: string]: unknown
}

/** A column the snapshot builder writes into the version's dataset.parquet, in order. */
export interface ColumnSpec {
  name: string
  /**
   * Tells the snapshot builder how to derive this column's value per item.
   * `split_index` and `item_id` are synthetic — appended by the snapshot
   * builder itself, not declared by any task in the registry (see
   * `SPLIT_INDEX_COLUMN`/`ITEM_ID_COLUMN` in lib/snapshot.ts) — so they're
   * never picked up by a task's own `inputFeatures`/`outputFeatures`. Ludwig
   * ignores parquet columns it wasn't told to use, so `item_id` rides along
   * unused during training and is only read back for evaluation reporting.
   * `text_sequence_label` reads an item's `text_sequence`-type annotation
   * (its `labelTextSequence` value) — the free-text ground truth for tasks
   * like image/audio captioning and ASR, as opposed to `label`, which only
   * ever resolves a `classification`-type annotation.
   */
  kind: 'storage_uri' | 'inline_text' | 'label' | 'text_sequence_label' | 'split' | 'split_index' | 'scalar' | 'item_id'
}

/** Everything the Ludwig feature builders need to know about one dataset version. */
export interface SnapshotContext {
  columns: ColumnSpec[]
  /** Label class names, in a stable order. Empty for non-classification tasks. */
  labelClassNames: string[]
  /**
   * Item count per class name, classification tasks only — lets the
   * compiler build balanced `class_weights` (see `useClassWeights` in
   * lib/ludwig/compile.ts) without a DB round trip at train-dispatch time.
   * Keyed by name, not id: Ludwig accepts a name-keyed `class_weights` dict
   * and resolves it against its own vocabulary internally, so this never
   * needs to know the index Ludwig assigns to each class.
   */
  classCounts?: Record<string, number>
}

export interface EncoderChoice {
  id: string
  label: string
  /** Ludwig encoder `type` value emitted into the compiled config. */
  encoderType: string
  pretrained: boolean
  /** Extra Ludwig encoder params merged in verbatim (e.g. torchvision `model_variant`). */
  params?: Record<string, unknown>
}

export interface TrainerKnobSpec {
  epochs: { default: number; min: number; max: number }
  batchSize: { default: number | 'auto'; options: (number | 'auto')[] }
  learningRate: { default: number; min: number; max: number }
  earlyStopPatience: { default: number; min: number }
}

export interface LudwigTaskConfig {
  modelType: 'ecd' | 'llm'
  inputFeatures: (ctx: SnapshotContext) => LudwigInputFeature[]
  outputFeatures: (ctx: SnapshotContext) => LudwigOutputFeature[]
  encoders: EncoderChoice[]
  trainerKnobs: TrainerKnobSpec
}

export interface TaskDescriptor {
  id: ProjectTask
  label: string
  modality: DatasetModality
  /** The extension seam: a non-Ludwig backend just needs a new value here plus its own worker. */
  backend: 'ludwig' | 'unsupported'
  status: 'stable' | 'experimental' | 'planned'
  /** What one dataset item carries. Drives upload, validation, and the pool UI. */
  itemSpec: { payload: 'file' | 'inline_text' | 'record'; accept?: string[] }
  /** Ground truth shape. Drives the Classes page and annotation editors. */
  annotation: { type: AnnotationType; requiresLabelClasses: boolean }
  /** Parquet columns the snapshot builder emits, in order. */
  columns: ColumnSpec[]
  /** Absent for `backend: 'unsupported'` tasks. */
  ludwig?: LudwigTaskConfig
}

/**
 * What a single inference request must supply for a task, derived from
 * `itemSpec.payload` plus (for `inline_text`) the task's own Ludwig input
 * feature column names — a multi-input task like question_answering needs
 * more than one text field. See `getInferenceInputSpec`.
 */
export type InferenceInputSpec =
  | { kind: 'file'; accept?: string[] }
  | { kind: 'text'; fields: string[] }
  | { kind: 'record' }

/** The shape a task's inference response takes — mirrors InferenceOutputSchema in server/lib/schema.ts. */
export type InferenceOutputKind = 'classification' | 'regression' | 'text' | 'tokens'

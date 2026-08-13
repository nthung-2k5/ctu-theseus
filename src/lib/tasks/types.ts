import type { AnnotationType, DatasetModality, ProjectTask } from '@server/lib/enums'

/** Ludwig ECD/LLM feature type strings, as they appear in a compiled config. */
export type LudwigFeatureType = 'image' | 'text' | 'audio' | 'number' | 'category' | 'binary' | 'sequence' | 'vector'

export interface LudwigInputFeature {
  name: string
  type: LudwigFeatureType
  column: string
  encoder?: { type: string; [key: string]: unknown }
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
  /** Tells the snapshot builder how to derive this column's value per item. */
  kind: 'storage_uri' | 'inline_text' | 'label' | 'split' | 'scalar'
}

/** Everything the Ludwig feature builders need to know about one dataset version. */
export interface SnapshotContext {
  columns: ColumnSpec[]
  /** Label class names, in a stable order. Empty for non-classification tasks. */
  labelClassNames: string[]
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

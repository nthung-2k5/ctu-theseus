/**
 * Decompiles a trained run's Ludwig artifacts into a Theseus-owned
 * `preprocessing.json` — the piece that makes a bare ONNX export usable.
 *
 * Ludwig's ONNX export is the raw graph: no resize/normalize, no
 * tokenization, no idx→label decode. Two sources feed the manifest:
 *
 *  - `training_runs.ludwigConfig` (Postgres) — the exact compiled config
 *    (feature names/types/columns), the symmetric counterpart to
 *    `lib/ludwig/compile.ts`.
 *  - `training_set_metadata.json` (S3, written by Ludwig at train time) —
 *    per-feature preprocessing params and, critically, `idx2str` for
 *    category outputs.
 *
 * `idx2str` MUST come from here, never from the `label_classes` table:
 * Postgres has no idea what index order Ludwig assigned a category's
 * classes, so guessing wrong here silently mislabels every prediction a
 * generated client makes.
 */

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { downloadFile, listKeys } from '@server/lib/storage'
import type { LudwigFeatureType } from '@server/lib/tasks'

export interface PreprocessingInput {
  name: string
  type: LudwigFeatureType
  column: string
  /**
   * Ludwig's raw per-feature `preprocessing` block from
   * training_set_metadata.json (image height/width/num_channels/
   * resize_method/standardize_image, text tokenizer params, ...), passed
   * through verbatim — not reinterpreted, except for `imageNormalization`
   * below where the preset name is a well-known public constant.
   */
  ludwigPreprocessing?: Record<string, unknown>
  /** Resolved only when `ludwigPreprocessing.standardize_image` matches a known preset (see IMAGE_STANDARDIZATION_PRESETS). */
  imageNormalization?: { mean: [number, number, number]; std: [number, number, number] }
  /**
   * `number`-type inputs only — the fitted normalization Ludwig computed at
   * train time (`numeric_transformation_registry` in Ludwig's
   * number_feature.py). The exported ONNX graph is the bare model: it has
   * no normalization built in (`NumberInputFeature.forward` goes straight
   * to the encoder), so a tabular client must replicate this exact
   * transform before feeding a raw value in, the same way the image client
   * replicates resize/normalize. `mean`/`std` are set for `zscore`,
   * `min`/`max` for `minmax`; `log1p` and `iq` (interquartile) need no
   * fitted params beyond their own type. Absent when normalization was
   * disabled (`"normalization": null`) or the input isn't type `number`.
   */
  numberNormalization?: {
    type: 'zscore' | 'minmax' | 'log1p' | 'iq'
    mean?: number
    std?: number
    min?: number
    max?: number
    q1?: number
    q2?: number
    q3?: number
  }
}

export interface PreprocessingOutput {
  name: string
  type: LudwigFeatureType
  column: string
  /** idx2str for `category` outputs, in Ludwig's internal index order. Absent for non-categorical outputs (e.g. regression). */
  classes?: string[]
}

export interface PreprocessingManifest {
  schemaVersion: 1
  runId: string
  inputs: PreprocessingInput[]
  outputs: PreprocessingOutput[]
}

/**
 * Standard ImageNet normalization constants (RGB mean/std, 0-1 scale) —
 * public, framework-agnostic values used by torchvision and virtually every
 * vision library, not something specific to Ludwig. Ludwig's
 * `standardize_image: "imagenet1k"` preprocessing preset applies exactly
 * these; unrecognized preset names are left as the raw string in
 * `ludwigPreprocessing` rather than guessed at.
 */
const IMAGE_STANDARDIZATION_PRESETS: Record<string, { mean: [number, number, number]; std: [number, number, number] }> =
  {
    imagenet1k: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
  }

interface CompiledFeature {
  name: string
  type: LudwigFeatureType
  column: string
  [key: string]: unknown
}

async function findTrainingSetMetadata(runId: string): Promise<Record<string, unknown> | null> {
  const keys = await listKeys(CONSTANTS.BUCKET_TRAINING, `${runId}/results/`)
  const metaKey = keys.find((k) => k.endsWith('training_set_metadata.json'))
  if (!metaKey) return null
  const bytes = await downloadFile(CONSTANTS.BUCKET_TRAINING, metaKey)
  return JSON.parse(new TextDecoder().decode(bytes))
}

export async function extractPreprocessing(runId: string): Promise<PreprocessingManifest> {
  const run = await db.query.trainingRuns.findFirst({ where: { id: runId } })
  if (!run) throw new Error(`Run ${runId} not found`)

  const ludwigConfig = run.ludwigConfig as {
    input_features?: CompiledFeature[]
    output_features?: CompiledFeature[]
  } | null
  if (!ludwigConfig?.input_features?.length) {
    throw new Error(`Run ${runId} has no compiled Ludwig config to decompile`)
  }

  const meta = await findTrainingSetMetadata(runId)

  const inputs: PreprocessingInput[] = ludwigConfig.input_features.map((f) => {
    const featureMeta = meta?.[f.name] as Record<string, unknown> | undefined
    const ludwigPreprocessing = featureMeta?.preprocessing as Record<string, unknown> | undefined
    const presetName = ludwigPreprocessing?.standardize_image
    const imageNormalization = typeof presetName === 'string' ? IMAGE_STANDARDIZATION_PRESETS[presetName] : undefined

    // Ludwig's fit_transform_params() writes mean/std (zscore) or min/max
    // (minmax) onto the feature's metadata dict directly — siblings of
    // `preprocessing`, not nested inside it (see number_feature.py's
    // get_feature_meta). `normalization` itself, the transform's own name,
    // IS inside `preprocessing`.
    const normalizationType = ludwigPreprocessing?.normalization as 'zscore' | 'minmax' | 'log1p' | 'iq' | undefined
    const numberNormalization =
      f.type === 'number' && normalizationType
        ? {
            type: normalizationType,
            mean: featureMeta?.mean as number | undefined,
            std: featureMeta?.std as number | undefined,
            min: featureMeta?.min as number | undefined,
            max: featureMeta?.max as number | undefined,
            q1: featureMeta?.q1 as number | undefined,
            q2: featureMeta?.q2 as number | undefined,
            q3: featureMeta?.q3 as number | undefined,
          }
        : undefined

    return { name: f.name, type: f.type, column: f.column, ludwigPreprocessing, imageNormalization, numberNormalization }
  })

  const outputs: PreprocessingOutput[] = (ludwigConfig.output_features ?? []).map((f) => {
    const featureMeta = meta?.[f.name] as Record<string, unknown> | undefined
    const idx2str = featureMeta?.idx2str as string[] | undefined
    return { name: f.name, type: f.type, column: f.column, classes: idx2str }
  })

  return { schemaVersion: 1, runId, inputs, outputs }
}

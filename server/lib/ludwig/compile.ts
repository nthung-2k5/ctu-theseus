import CONSTANTS from '@schema/constants.json'
import type {
  EncoderChoice,
  ImageAugmentationType,
  LudwigInputFeature,
  LudwigOptimizerType,
  LudwigOutputFeature,
  SnapshotContext,
  TaskDescriptor,
} from '@server/lib/tasks'
import { ImageAugmentationTypes, LudwigOptimizerTypes } from '@server/lib/tasks'
import { type LudwigConfig, ludwigConfigSchema } from './schema'

export interface TrainerSelections {
  epochs?: number
  batchSize?: number | 'auto'
  learningRate?: number
  earlyStopPatience?: number
  /** Encoder id from the task's TaskDescriptor.ludwig.encoders list. Defaults to the first. */
  encoderId?: string
  /**
   * Weight each class's loss contribution inversely to its frequency in the
   * snapshot (see `applyClassWeights` below) — for a category output only;
   * ignored for every other output type. Requires the snapshot manifest to
   * carry `classCounts` (see server/lib/snapshot.ts's `buildSnapshot`).
   */
  useClassWeights?: boolean
  /** Ludwig image augmentation ops to apply to every `image`-type input feature. Ignored for non-vision tasks. */
  augmentations?: ImageAugmentationType[]
  /** Square-resize every `image`-type input feature to this many pixels per side before training. Ignored for non-vision tasks. */
  imageSize?: number
  /**
   * Metric Ludwig tracks for early stopping / "best epoch" instead of its
   * own per-output-type default (e.g. `loss`, `accuracy`, `mean_squared_error`)
   * — passed straight through as `trainer.validation_metric`; Ludwig itself
   * rejects a metric name the model's output feature type doesn't support.
   */
  validationMetric?: string
  /** Ludwig optimizer type. Defaults to Ludwig's own per-model-type default (Adam for ECD) when unset. */
  optimizer?: LudwigOptimizerType
}

const LUDWIG_VERSION = '0.17.5'

export function compileLudwigConfig(
  task: TaskDescriptor,
  ctx: SnapshotContext,
  selections: TrainerSelections = {},
): LudwigConfig {
  if (!task.ludwig) {
    throw new Error(`Task "${task.id}" has no Ludwig backend (status: ${task.status})`)
  }
  const { ludwig } = task
  const knobs = ludwig.trainerKnobs

  const declaredInputFeatures = ludwig.inputFeatures(ctx)
  const inputFeatures: LudwigInputFeature[] =
    declaredInputFeatures.length > 0
      ? declaredInputFeatures.map((f) =>
          withImageResize(
            withAugmentation(withEncoder(f, ludwig.encoders, selections.encoderId), selections.augmentations),
            selections.imageSize,
          ),
        )
      : // Tabular tasks don't know their column names statically — derive one
        // `number` feature per scalar column the snapshot actually contains.
        ctx.columns
          .filter((c) => c.kind === 'scalar')
          .map((c) => ({ name: c.name, type: 'number' as const, column: c.name }))

  if (inputFeatures.length === 0) {
    throw new Error(`Task "${task.id}": no input features could be derived from the snapshot's columns`)
  }

  const outputFeatures = selections.useClassWeights
    ? applyClassWeights(ludwig.outputFeatures(ctx), ctx.classCounts, task.id)
    : ludwig.outputFeatures(ctx)

  const config: LudwigConfig = {
    model_type: ludwig.modelType,
    input_features: inputFeatures,
    output_features: outputFeatures,
    // Without this, Ludwig defaults to a random 70/10/20 re-split and
    // ignores the split the user actually assigned (manually or via
    // POST /items/auto-split) — see server/lib/snapshot.ts for why this
    // must be the synthetic integer column, not the human-readable one.
    preprocessing: {
      split: { type: 'fixed', column: CONSTANTS.SPLIT_INDEX_COLUMN_NAME },
    },
    trainer: {
      epochs: selections.epochs ?? knobs.epochs.default,
      batch_size: selections.batchSize ?? knobs.batchSize.default,
      learning_rate: selections.learningRate ?? knobs.learningRate.default,
      early_stop: selections.earlyStopPatience ?? knobs.earlyStopPatience.default,
      ...(selections.validationMetric && { validation_metric: selections.validationMetric }),
      ...(selections.optimizer && { optimizer: withOptimizer(selections.optimizer, task.id) }),
    },
    ludwig_version: LUDWIG_VERSION,
  }

  // Validated at compile time, before the run row is ever committed — an
  // invalid configuration surfaces as a 400 here, not as a worker crash
  // discovered ten seconds later.
  return ludwigConfigSchema.parse(config)
}

function withEncoder(
  feature: LudwigInputFeature,
  encoders: EncoderChoice[],
  encoderId: string | undefined,
): LudwigInputFeature {
  if (feature.encoder || !encoders || encoders.length === 0) return feature

  const encoder = encoderId ? encoders.find((e) => e.id === encoderId) : encoders[0]
  if (!encoder) {
    const available = encoders.map((e) => e.id).join(', ')
    throw new Error(`Unknown encoder "${encoderId}" (available: ${available})`)
  }

  return {
    ...feature,
    encoder: { type: encoder.encoderType, use_pretrained: encoder.pretrained, ...encoder.params },
  }
}

/**
 * Attach Ludwig's built-in image augmentation ops to an `image`-type input
 * feature (ludwig.schema.features.augmentation.image — each op's own
 * defaults apply since only `type` is set here). A no-op for every other
 * feature type, so this is safe to call unconditionally over a task's whole
 * input feature list regardless of modality.
 */
function withAugmentation(
  feature: LudwigInputFeature,
  augmentations: ImageAugmentationType[] | undefined,
): LudwigInputFeature {
  if (!augmentations || augmentations.length === 0 || feature.type !== 'image') return feature

  const unknown = augmentations.find((a) => !ImageAugmentationTypes.includes(a))
  if (unknown) {
    throw new Error(`Unknown augmentation "${unknown}" (available: ${ImageAugmentationTypes.join(', ')})`)
  }

  return { ...feature, augmentation: augmentations.map((type) => ({ type })) }
}

/**
 * Square-resize an `image`-type input feature to `size` pixels per side
 * (Ludwig's `preprocessing.height`/`width`, ludwig.schema.features.
 * preprocessing.image) before training. A no-op for every other feature
 * type or when `size` is unset, so safe to call unconditionally.
 */
function withImageResize(feature: LudwigInputFeature, size: number | undefined): LudwigInputFeature {
  if (!size || feature.type !== 'image') return feature
  return { ...feature, preprocessing: { ...feature.preprocessing, height: size, width: size } }
}

/** Validates the optimizer choice against the known-safe subset this app exposes (see LudwigOptimizerTypes). */
function withOptimizer(optimizer: LudwigOptimizerType, taskId: string): { type: LudwigOptimizerType } {
  if (!LudwigOptimizerTypes.includes(optimizer)) {
    throw new Error(`Task "${taskId}": unknown optimizer "${optimizer}" (available: ${LudwigOptimizerTypes.join(', ')})`)
  }
  return { type: optimizer }
}

/**
 * Balanced inverse-frequency weights for a `category` output feature's loss,
 * keyed by class NAME rather than index — Ludwig resolves a name-keyed
 * `class_weights` dict against its own internally-built vocabulary
 * (`str2idx`) at preprocessing time (see `category_feature.py`'s
 * `update_config_after_module_init`), so this never has to know or predict
 * the index Ludwig will assign each class. Predicting that index ourselves
 * would be fragile: Ludwig builds it from the *training data's own value
 * frequencies* (most-common-first), not any order available before training
 * runs.
 *
 * `weight = totalCount / (numClasses * classCount)` — the standard
 * "balanced" formula (equivalent to scikit-learn's `class_weight="balanced"`):
 * an underrepresented class gets a weight above 1, an overrepresented one
 * below 1, and a perfectly balanced dataset gets all-1s.
 */
function applyClassWeights(
  outputFeatures: LudwigOutputFeature[],
  classCounts: Record<string, number> | undefined,
  taskId: string,
): LudwigOutputFeature[] {
  if (!classCounts || Object.keys(classCounts).length === 0) {
    throw new Error(`Task "${taskId}": class weighting requires a snapshot with a recorded class distribution`)
  }
  const totalCount = Object.values(classCounts).reduce((sum, count) => sum + count, 0)
  const numClasses = Object.keys(classCounts).length
  const weights = Object.fromEntries(
    Object.entries(classCounts).map(([name, count]) => [name, totalCount / (numClasses * count)]),
  )

  return outputFeatures.map((feature) =>
    feature.type === 'category'
      ? { ...feature, loss: { type: 'softmax_cross_entropy', class_weights: weights } }
      : feature,
  )
}

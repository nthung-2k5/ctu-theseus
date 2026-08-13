import type { EncoderChoice, LudwigInputFeature, SnapshotContext, TaskDescriptor } from '@server/lib/tasks'
import { type LudwigConfig, ludwigConfigSchema } from './schema'

export interface TrainerSelections {
  epochs?: number
  batchSize?: number | 'auto'
  learningRate?: number
  earlyStopPatience?: number
  /** Encoder id from the task's TaskDescriptor.ludwig.encoders list. Defaults to the first. */
  encoderId?: string
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
      ? declaredInputFeatures.map((f) => withEncoder(f, ludwig.encoders, selections.encoderId))
      : // Tabular tasks don't know their column names statically — derive one
        // `number` feature per scalar column the snapshot actually contains.
        ctx.columns
          .filter((c) => c.kind === 'scalar')
          .map((c) => ({ name: c.name, type: 'number' as const, column: c.name }))

  if (inputFeatures.length === 0) {
    throw new Error(`Task "${task.id}": no input features could be derived from the snapshot's columns`)
  }

  const config: LudwigConfig = {
    model_type: ludwig.modelType,
    input_features: inputFeatures,
    output_features: ludwig.outputFeatures(ctx),
    trainer: {
      epochs: selections.epochs ?? knobs.epochs.default,
      batch_size: selections.batchSize ?? knobs.batchSize.default,
      learning_rate: selections.learningRate ?? knobs.learningRate.default,
      early_stop: selections.earlyStopPatience ?? knobs.earlyStopPatience.default,
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

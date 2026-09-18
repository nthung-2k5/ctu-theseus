import { z } from 'zod'

const featureType = z.enum(['image', 'text', 'audio', 'number', 'category', 'binary', 'sequence', 'vector'])

const encoderSchema = z.object({ type: z.string() }).catchall(z.unknown())
const lossSchema = z.object({ type: z.string() }).catchall(z.unknown())
const optimizerSchema = z.object({ type: z.string() }).catchall(z.unknown())

const inputFeatureSchema = z
  .object({
    name: z.string(),
    type: featureType,
    column: z.string(),
    encoder: encoderSchema.optional(),
  })
  .catchall(z.unknown())

const outputFeatureSchema = z
  .object({
    name: z.string(),
    type: featureType,
    column: z.string(),
    loss: lossSchema.optional(),
  })
  .catchall(z.unknown())

const trainerSchema = z.object({
  epochs: z.number().int().positive(),
  batch_size: z.union([z.number().int().positive(), z.literal('auto')]),
  learning_rate: z.number().positive(),
  early_stop: z.number().int(),
  // Both optional — declared here (rather than left to `trainer`'s object
  // shape alone) because a plain z.object silently STRIPS undeclared keys
  // on parse, unlike the `.catchall(z.unknown())` schemas above. Omitting
  // these would make compileLudwigConfig's `useValidationMetric`/
  // `useOptimizer` selections vanish from the config actually sent to the
  // worker, with no error anywhere to catch it.
  validation_metric: z.string().optional(),
  optimizer: optimizerSchema.optional(),
})

/**
 * Pins the split Ludwig actually trains on to the split the user assigned
 * (see server/lib/snapshot.ts's `_ludwig_split_idx` column) rather than
 * Ludwig's default random re-partition, which would silently ignore the
 * dataset's train/validation/test assignment (and the auto-split ratios)
 * entirely.
 */
const preprocessingSchema = z.object({
  split: z.object({
    type: z.literal('fixed'),
    column: z.string(),
  }),
})

export const ludwigConfigSchema = z.object({
  model_type: z.enum(['ecd', 'llm']),
  input_features: z.array(inputFeatureSchema).min(1),
  output_features: z.array(outputFeatureSchema).min(1),
  preprocessing: preprocessingSchema,
  trainer: trainerSchema,
  ludwig_version: z.string(),
})

export type LudwigConfig = z.infer<typeof ludwigConfigSchema>

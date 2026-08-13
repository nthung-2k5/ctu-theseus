import { z } from 'zod'

const featureType = z.enum(['image', 'text', 'audio', 'number', 'category', 'binary', 'sequence', 'vector'])

const encoderSchema = z.object({ type: z.string() }).catchall(z.unknown())
const lossSchema = z.object({ type: z.string() }).catchall(z.unknown())

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
})

export const ludwigConfigSchema = z.object({
  model_type: z.enum(['ecd', 'llm']),
  input_features: z.array(inputFeatureSchema).min(1),
  output_features: z.array(outputFeatureSchema).min(1),
  trainer: trainerSchema,
  ludwig_version: z.string(),
})

export type LudwigConfig = z.infer<typeof ludwigConfigSchema>

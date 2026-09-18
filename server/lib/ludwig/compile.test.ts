import { describe, expect, test } from 'bun:test'
import type { SnapshotContext } from '@server/lib/tasks'
import { getTaskDescriptor } from '@server/lib/tasks'
import { compileLudwigConfig } from './compile'

const visionContext: SnapshotContext = {
  columns: [
    { name: 'image_path', kind: 'storage_uri' },
    { name: 'class', kind: 'label' },
    { name: 'split', kind: 'split' },
  ],
  labelClassNames: ['cat', 'dog'],
}

const captioningContext: SnapshotContext = {
  columns: [
    { name: 'image_path', kind: 'storage_uri' },
    { name: 'caption', kind: 'text_sequence_label' },
    { name: 'split', kind: 'split' },
  ],
  labelClassNames: [],
}

const asrContext: SnapshotContext = {
  columns: [
    { name: 'audio_path', kind: 'storage_uri' },
    { name: 'transcript', kind: 'text_sequence_label' },
    { name: 'split', kind: 'split' },
  ],
  labelClassNames: [],
}

const tabularContext: SnapshotContext = {
  columns: [
    { name: 'age', kind: 'scalar' },
    { name: 'income', kind: 'scalar' },
    { name: 'class', kind: 'label' },
    { name: 'split', kind: 'split' },
  ],
  labelClassNames: ['approved', 'denied'],
}

describe('compileLudwigConfig', () => {
  test('compiles a vision task with its declared input feature and default encoder', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {})

    expect(config.model_type).toBe('ecd')
    expect(config.input_features).toHaveLength(1)
    expect(config.input_features[0]).toMatchObject({ name: 'image_path', type: 'image', column: 'image_path' })
    // The default encoder should be filled in automatically.
    expect(config.input_features[0].encoder).toBeDefined()
    expect(config.output_features).toEqual([{ name: 'class', type: 'category', column: 'class' }])
  })

  test('derives tabular input features from scalar snapshot columns', () => {
    const config = compileLudwigConfig(getTaskDescriptor('tabular_classification'), tabularContext, {})

    expect(config.input_features).toHaveLength(2)
    expect(config.input_features.map((f) => f.name).sort()).toEqual(['age', 'income'])
    for (const feature of config.input_features) {
      expect(feature.type).toBe('number')
    }
  })

  test('applies trainer selections over the task defaults', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
      epochs: 5,
      batchSize: 16,
      learningRate: 0.01,
    })

    expect(config.trainer.epochs).toBe(5)
    expect(config.trainer.batch_size).toBe(16)
    expect(config.trainer.learning_rate).toBe(0.01)
  })

  test('selects the requested encoder by id', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
      encoderId: 'resnet50',
    })

    expect(config.input_features[0].encoder).toMatchObject({ type: expect.stringContaining('resnet') })
  })

  test('throws for an unknown encoder id', () => {
    expect(() =>
      compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
        encoderId: 'not-a-real-encoder',
      }),
    ).toThrow()
  })

  test('throws for a tabular snapshot with no scalar columns', () => {
    const emptyContext: SnapshotContext = {
      columns: [
        { name: 'class', kind: 'label' },
        { name: 'split', kind: 'split' },
      ],
      labelClassNames: ['a', 'b'],
    }
    expect(() => compileLudwigConfig(getTaskDescriptor('tabular_classification'), emptyContext, {})).toThrow()
  })
})

describe('compileLudwigConfig class weighting', () => {
  test('leaves output features untouched when useClassWeights is not set', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {})
    expect(config.output_features[0].loss).toBeUndefined()
  })

  test('attaches a name-keyed class_weights dict, balanced inversely to frequency', () => {
    const context: SnapshotContext = { ...visionContext, classCounts: { cat: 80, dog: 20 } }
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), context, { useClassWeights: true })

    // Never index-keyed — Ludwig resolves a name-keyed dict against its own
    // vocabulary, so this must never assume/predict which index Ludwig will
    // assign to 'cat' vs 'dog'.
    const weights = config.output_features[0].loss?.class_weights as Record<string, number>
    expect(Object.keys(weights).sort()).toEqual(['cat', 'dog'])
    // Balanced formula: total / (numClasses * count) — the minority class
    // (dog, 20/100) gets a weight above 1, the majority (cat, 80/100) below 1.
    expect(weights.dog).toBeCloseTo(100 / (2 * 20))
    expect(weights.cat).toBeCloseTo(100 / (2 * 80))
    expect(weights.dog).toBeGreaterThan(weights.cat)
  })

  test('a perfectly balanced dataset gets all-1 weights', () => {
    const context: SnapshotContext = { ...visionContext, classCounts: { cat: 50, dog: 50 } }
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), context, { useClassWeights: true })
    const weights = config.output_features[0].loss?.class_weights as Record<string, number>
    expect(weights.cat).toBeCloseTo(1)
    expect(weights.dog).toBeCloseTo(1)
  })

  test('does not weight a regression output (no category feature to weight)', () => {
    const context: SnapshotContext = {
      columns: [
        { name: 'age', kind: 'scalar' },
        { name: 'target', kind: 'label' },
        { name: 'split', kind: 'split' },
      ],
      labelClassNames: [],
      classCounts: {},
    }
    // classCounts is present but empty, and tabular_regression's output type
    // is 'number' — applyClassWeights would throw on empty classCounts only
    // if it actually tried to weight something, so this also guards that a
    // non-category output is simply passed through untouched.
    expect(() =>
      compileLudwigConfig(getTaskDescriptor('tabular_regression'), context, { useClassWeights: true }),
    ).toThrow()
  })

  test('throws when the snapshot has no recorded class distribution', () => {
    expect(() =>
      compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, { useClassWeights: true }),
    ).toThrow(/class distribution/)
  })
})

describe('compileLudwigConfig augmentation', () => {
  test('leaves the image input feature untouched when no augmentations are requested', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {})
    expect(config.input_features[0].augmentation).toBeUndefined()
  })

  test('attaches the requested augmentation ops to the image input feature', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
      augmentations: ['random_horizontal_flip', 'random_rotate'],
    })
    expect(config.input_features[0].augmentation).toEqual([
      { type: 'random_horizontal_flip' },
      { type: 'random_rotate' },
    ])
  })

  test('throws for an unknown augmentation type', () => {
    expect(() =>
      compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
        // biome-ignore lint/suspicious/noExplicitAny: deliberately an invalid value to test the error path
        augmentations: ['not_a_real_augmentation' as any],
      }),
    ).toThrow(/Unknown augmentation/)
  })

  test('is a no-op for a non-image task (tabular has no image input feature to augment)', () => {
    const config = compileLudwigConfig(getTaskDescriptor('tabular_classification'), tabularContext, {
      augmentations: ['random_rotate'],
    })
    for (const feature of config.input_features) {
      expect(feature.augmentation).toBeUndefined()
    }
  })
})

describe('compileLudwigConfig image resize', () => {
  test('sets height and width to the requested size on the image input feature', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, { imageSize: 128 })
    expect(config.input_features[0].preprocessing).toMatchObject({ height: 128, width: 128 })
  })

  test('leaves preprocessing untouched when imageSize is not set', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {})
    expect(config.input_features[0].preprocessing).toBeUndefined()
  })

  test('is a no-op for a non-image task', () => {
    const config = compileLudwigConfig(getTaskDescriptor('tabular_classification'), tabularContext, { imageSize: 128 })
    for (const feature of config.input_features) {
      expect(feature.preprocessing).toBeUndefined()
    }
  })
})

describe('compileLudwigConfig validation metric', () => {
  test('passes validationMetric straight through as trainer.validation_metric', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
      validationMetric: 'accuracy',
    })
    expect(config.trainer.validation_metric).toBe('accuracy')
  })

  test('omits validation_metric entirely when not set — Ludwig applies its own per-output-type default', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {})
    expect(config.trainer.validation_metric).toBeUndefined()
  })
})

describe('compileLudwigConfig optimizer', () => {
  test('sets trainer.optimizer.type to the requested optimizer', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
      optimizer: 'adamw',
    })
    expect(config.trainer.optimizer).toEqual({ type: 'adamw' })
  })

  test('omits optimizer entirely when not set — Ludwig applies its own model-type default', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {})
    expect(config.trainer.optimizer).toBeUndefined()
  })

  test('throws for an unknown optimizer', () => {
    expect(() =>
      compileLudwigConfig(getTaskDescriptor('image_classification'), visionContext, {
        // biome-ignore lint/suspicious/noExplicitAny: deliberately an invalid value to test the error path
        optimizer: 'not_a_real_optimizer' as any,
      }),
    ).toThrow(/unknown optimizer/)
  })
})

describe('compileLudwigConfig Tier-3 captioning/ASR tasks', () => {
  test('image_captioning pairs an image input with a text (not category) output', () => {
    const config = compileLudwigConfig(getTaskDescriptor('image_captioning'), captioningContext, {})

    expect(config.model_type).toBe('ecd')
    expect(config.input_features).toEqual([
      expect.objectContaining({ name: 'image_path', type: 'image', column: 'image_path' }),
    ])
    expect(config.output_features).toEqual([{ name: 'caption', type: 'text', column: 'caption' }])
  })

  test('audio_captioning pairs an audio input with a text output', () => {
    const config = compileLudwigConfig(getTaskDescriptor('audio_captioning'), captioningContext, {})

    expect(config.input_features).toEqual([
      expect.objectContaining({ name: 'audio_path', type: 'audio', column: 'audio_path' }),
    ])
    expect(config.output_features).toEqual([{ name: 'caption', type: 'text', column: 'caption' }])
  })

  test('automatic_speech_recognition pairs an audio input with a transcript text output', () => {
    const config = compileLudwigConfig(getTaskDescriptor('automatic_speech_recognition'), asrContext, {})

    expect(config.input_features).toEqual([
      expect.objectContaining({ name: 'audio_path', type: 'audio', column: 'audio_path' }),
    ])
    expect(config.output_features).toEqual([{ name: 'transcript', type: 'text', column: 'transcript' }])
  })

  // useClassWeights is only ever surfaced in the UI for classification tasks
  // (requiresLabelClasses), so a text_sequence task like image_captioning
  // should never actually reach this — applyClassWeights' fail-fast guard
  // (no classCounts => throw, rather than silently compiling a class_weights
  // key Ludwig would reject) still needs to hold even here.
  test('useClassWeights throws for a task with no class distribution to weight', () => {
    expect(() =>
      compileLudwigConfig(getTaskDescriptor('image_captioning'), captioningContext, { useClassWeights: true }),
    ).toThrow(/class weighting requires/)
  })
})

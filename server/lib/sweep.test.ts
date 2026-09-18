import { describe, expect, test } from 'bun:test'
import { expandGrid, expandSweep, sampleRandom, validateSearchSpace } from './sweep'

describe('validateSearchSpace', () => {
  test('rejects an empty search space', () => {
    expect(validateSearchSpace({}, 5)).toMatch(/at least one hyperparameter/)
  })

  test('rejects a knob with no candidate values', () => {
    expect(validateSearchSpace({ learningRate: [] }, 5)).toMatch(/learningRate/)
  })

  test('rejects maxTrials outside [1, 50]', () => {
    expect(validateSearchSpace({ learningRate: [0.01] }, 0)).toMatch(/maxTrials/)
    expect(validateSearchSpace({ learningRate: [0.01] }, 51)).toMatch(/maxTrials/)
  })

  test('accepts a valid search space', () => {
    expect(validateSearchSpace({ learningRate: [0.01, 0.001] }, 4)).toBeNull()
  })
})

describe('expandGrid', () => {
  test('produces the full cartesian product', () => {
    const trials = expandGrid({ learningRate: [0.01, 0.001], batchSize: [16, 32] })
    expect(trials).toHaveLength(4)
    expect(trials).toEqual(
      expect.arrayContaining([
        { learningRate: 0.01, batchSize: 16 },
        { learningRate: 0.01, batchSize: 32 },
        { learningRate: 0.001, batchSize: 16 },
        { learningRate: 0.001, batchSize: 32 },
      ]),
    )
  })

  test('a single-knob search space produces one trial per candidate', () => {
    const trials = expandGrid({ encoderId: ['resnet18', 'resnet50', 'vit_base'] })
    expect(trials).toEqual([{ encoderId: 'resnet18' }, { encoderId: 'resnet50' }, { encoderId: 'vit_base' }])
  })

  test('an empty search space produces exactly one (empty) trial', () => {
    expect(expandGrid({})).toEqual([{}])
  })
})

describe('sampleRandom', () => {
  test('produces exactly `count` trials, each drawn from the declared candidates', () => {
    const searchSpace = { learningRate: [0.01, 0.001], encoderId: ['resnet18', 'resnet50'] }
    const trials = sampleRandom(searchSpace, 10)
    expect(trials).toHaveLength(10)
    for (const trial of trials) {
      expect(trial.learningRate).toBeDefined()
      expect(trial.encoderId).toBeDefined()
      expect(searchSpace.learningRate).toContain(trial.learningRate as number)
      expect(searchSpace.encoderId).toContain(trial.encoderId as string)
    }
  })

  test('count 0 produces no trials', () => {
    expect(sampleRandom({ learningRate: [0.01] }, 0)).toEqual([])
  })
})

describe('expandSweep', () => {
  test('grid truncates to maxTrials rather than sampling down', () => {
    const trials = expandSweep({ batchSize: [16, 32, 64, 128] }, 'grid', 2)
    expect(trials).toEqual([{ batchSize: 16 }, { batchSize: 32 }])
  })

  test('grid returns the full product when maxTrials exceeds it', () => {
    const trials = expandSweep({ batchSize: [16, 32] }, 'grid', 10)
    expect(trials).toHaveLength(2)
  })

  test('random always returns exactly maxTrials trials', () => {
    const trials = expandSweep({ batchSize: [16, 32] }, 'random', 7)
    expect(trials).toHaveLength(7)
  })
})

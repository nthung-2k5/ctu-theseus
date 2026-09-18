import { describe, expect, test } from 'bun:test'
import { STREAM_DEFS, SUBJECT_TEMPLATES, subject } from './subjects'

describe('subject builders', () => {
  test('fill in run/job ids at the trailing token', () => {
    expect(subject.trainTask('run-1')).toBe('theseus.task.train.run-1')
    expect(subject.exportTask('job-1')).toBe('theseus.task.export.job-1')
    expect(subject.abortCommand('run-1')).toBe('theseus.command.run.run-1')
    expect(subject.inferenceTask('inf-1')).toBe('theseus.task.inference.inf-1')
    expect(subject.inferenceWarm('run-1')).toBe('theseus.inference.warm.run-1')
    expect(subject.inferenceResult('inf-1')).toBe('theseus.inference.result.inf-1')
    expect(subject.abortFlag('run-1')).toBe('theseus.abortflag.run-1')
  })

  test('runEvent fills both the run id and the event kind', () => {
    expect(subject.runEvent('run-1', 'status')).toBe('theseus.event.run.run-1.status')
    expect(subject.runEventsWildcard('run-1')).toBe('theseus.event.run.run-1.>')
  })

  test('dlq fills both the kind and id tokens', () => {
    expect(subject.dlq('train-worker', 'run-1')).toBe('theseus.dlq.train-worker.run-1')
  })

  test('every template placeholder is consumed by its builder (no stray {token} left in output)', () => {
    for (const built of [
      subject.trainTask('x'),
      subject.exportTask('x'),
      subject.abortCommand('x'),
      subject.runEvent('x', 'status'),
      subject.runEventsWildcard('x'),
      subject.inferenceTask('x'),
      subject.inferenceWarm('x'),
      subject.inferenceResult('x'),
      subject.dlq('x', 'y'),
      subject.abortFlag('x'),
    ]) {
      expect(built).not.toMatch(/\{[a-zA-Z]+\}/)
    }
  })

  test('SUBJECT_TEMPLATES has one entry per subject-builder key', () => {
    expect(Object.keys(SUBJECT_TEMPLATES).sort()).toEqual(Object.keys(subject).sort())
  })
})

describe('STREAM_DEFS', () => {
  test('THESEUS_ABORT_FLAGS and THESEUS_INFERENCE_RESULTS are the only last-value-per-subject streams', () => {
    const lastValueStreams = ['THESEUS_ABORT_FLAGS', 'THESEUS_INFERENCE_RESULTS']
    for (const name of lastValueStreams) {
      expect(STREAM_DEFS.find((s) => s.name === name)?.maxMsgsPerSubject).toBe(1)
    }

    for (const def of STREAM_DEFS) {
      if (!lastValueStreams.includes(def.name)) expect(def.maxMsgsPerSubject).toBeUndefined()
    }
  })

  test('every stream name is unique', () => {
    const names = STREAM_DEFS.map((s) => s.name)
    expect(new Set(names).size).toBe(names.length)
  })
})

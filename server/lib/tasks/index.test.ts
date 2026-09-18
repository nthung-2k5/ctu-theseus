import { describe, expect, test } from 'bun:test'
import { getInferenceInputSpec, getInferenceOutputKind } from './index'

describe('getInferenceInputSpec', () => {
  test('file-backed tasks report their accepted MIME types', () => {
    expect(getInferenceInputSpec('image_classification')).toEqual({
      kind: 'file',
      accept: ['image/jpeg', 'image/png'],
    })
  })

  test('record-backed tasks carry no fixed field list — the dataset defines the columns', () => {
    expect(getInferenceInputSpec('tabular_classification')).toEqual({ kind: 'record' })
    expect(getInferenceInputSpec('tabular_regression')).toEqual({ kind: 'record' })
  })

  test('single-input text tasks report one field named after their Ludwig input column', () => {
    expect(getInferenceInputSpec('text_classification')).toEqual({ kind: 'text', fields: ['text'] })
    expect(getInferenceInputSpec('text_generation')).toEqual({ kind: 'text', fields: ['prompt'] })
    expect(getInferenceInputSpec('summarization')).toEqual({ kind: 'text', fields: ['document'] })
    expect(getInferenceInputSpec('sequence_to_sequence')).toEqual({ kind: 'text', fields: ['source'] })
  })

  test('question_answering reports both its input fields, not just the first', () => {
    expect(getInferenceInputSpec('question_answering')).toEqual({ kind: 'text', fields: ['context', 'question'] })
  })
})

describe('getInferenceOutputKind', () => {
  test('category outputs are classification', () => {
    expect(getInferenceOutputKind('image_classification')).toBe('classification')
    expect(getInferenceOutputKind('text_classification')).toBe('classification')
    expect(getInferenceOutputKind('tabular_classification')).toBe('classification')
  })

  test('number outputs are regression, not a confidence score', () => {
    expect(getInferenceOutputKind('tabular_regression')).toBe('regression')
  })

  test('sequence outputs are tokens', () => {
    expect(getInferenceOutputKind('token_classification')).toBe('tokens')
  })

  test('generated-text outputs, including multi-input question_answering, are text', () => {
    expect(getInferenceOutputKind('text_generation')).toBe('text')
    expect(getInferenceOutputKind('summarization')).toBe('text')
    expect(getInferenceOutputKind('sequence_to_sequence')).toBe('text')
    expect(getInferenceOutputKind('question_answering')).toBe('text')
  })
})

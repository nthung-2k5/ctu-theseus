import { describe, expect, test } from 'bun:test'
import type { ColumnSpec } from '@server/lib/tasks'
import { type MemberWithItem, resolveColumnValue } from './snapshot'

const baseMember: MemberWithItem = {
  versionId: 'version-1',
  itemId: 'item-1',
  splitType: 'train',
  item: {
    storageUrl: 'pool/abc.png',
    textFeatures: { rawText: 'hello world' },
    tabularFeatures: { featuresJson: { age: 42 } },
    annotations: [
      { classId: 'class-1', labelStructured: null, annotationType: 'classification', labelTextSequence: null },
    ],
  },
}

const captionedMember: MemberWithItem = {
  ...baseMember,
  item: {
    ...baseMember.item,
    annotations: [
      { classId: null, labelStructured: null, annotationType: 'text_sequence', labelTextSequence: 'a cat sitting on a mat' },
    ],
  },
}

const classNameById = new Map([['class-1', 'cat']])

describe('resolveColumnValue', () => {
  test('item_id resolves to the pool item id, unaffected by any other column kind', () => {
    const col: ColumnSpec = { name: '_theseus_item_id', kind: 'item_id' }
    expect(resolveColumnValue(col, baseMember, classNameById)).toBe('item-1')
  })

  test('existing column kinds are unchanged by the item_id addition', () => {
    expect(resolveColumnValue({ name: 'split', kind: 'split' }, baseMember, classNameById)).toBe('train')
    expect(resolveColumnValue({ name: 'idx', kind: 'split_index' }, baseMember, classNameById)).toBe(0)
    expect(resolveColumnValue({ name: 'class', kind: 'label' }, baseMember, classNameById)).toBe('cat')
    expect(resolveColumnValue({ name: 'text', kind: 'inline_text' }, baseMember, classNameById)).toBe('hello world')
  })

  test('text_sequence_label resolves a text_sequence annotation, not a classification one', () => {
    const col: ColumnSpec = { name: 'caption', kind: 'text_sequence_label' }
    expect(resolveColumnValue(col, captionedMember, classNameById)).toBe('a cat sitting on a mat')
    // baseMember's only annotation is classification-type, so there's no text_sequence to find.
    expect(resolveColumnValue(col, baseMember, classNameById)).toBeNull()
  })
})

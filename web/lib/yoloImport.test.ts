import { expect, test } from 'bun:test'
import {
  chunkBySplit,
  isImageName,
  isLabelName,
  isNamesFileName,
  pairKey,
  parseLabelFile,
  parseNamesFile,
  parseYamlNames,
  planImport,
  splitOfPath,
  uploadBoxes,
  type YoloEntry,
} from './yoloImport'

const file = (name: string, size = 10) => new File([new Uint8Array(size)], name)
const entry = (...path: string[]): YoloEntry => ({ path, file: file(path[path.length - 1]) })

test('file kinds: jpeg and png are images, a .txt is a label unless it is the class list', () => {
  expect(isImageName('a.JPG')).toBe(true)
  expect(isImageName('a.jpeg')).toBe(true)
  expect(isImageName('a.png')).toBe(true)
  expect(isImageName('a.webp')).toBe(false) // the server only takes jpeg and png
  expect(isLabelName('a.txt')).toBe(true)
  expect(isLabelName('classes.txt')).toBe(false)
  expect(isNamesFileName('Data.YAML')).toBe(true)
  expect(isNamesFileName('a.txt')).toBe(false)
})

test('an image and its label pair up through the images/labels swap, and by path when they sit side by side', () => {
  expect(pairKey(['ds', 'images', 'train', 'cat 1.jpg'])).toBe(pairKey(['ds', 'labels', 'train', 'cat 1.txt']))
  expect(pairKey(['a.jpg'])).toBe(pairKey(['a.txt']))
  expect(pairKey(['set', 'x.png'])).toBe(pairKey(['set', 'x.txt']))
  // Not each other's: a different split, a different name, a different dataset folder.
  expect(pairKey(['ds', 'images', 'train', 'a.jpg'])).not.toBe(pairKey(['ds', 'labels', 'val', 'a.txt']))
  expect(pairKey(['ds', 'images', 'train', 'a.jpg'])).not.toBe(pairKey(['ds', 'labels', 'train', 'b.txt']))
  expect(pairKey(['one', 'images', 'a.jpg'])).not.toBe(pairKey(['two', 'labels', 'a.txt']))
  // Only the LAST images folder is the marker, as in Ultralytics: a dataset that lives inside a folder called images.
  expect(pairKey(['images', 'ds', 'images', 'a.jpg'])).toBe(pairKey(['images', 'ds', 'labels', 'a.txt']))
})

test('the split comes from the nearest folder that names one, else nothing', () => {
  expect(splitOfPath(['ds', 'images', 'train', 'a.jpg'])).toBe('train')
  expect(splitOfPath(['images', 'val', 'a.jpg'])).toBe('validation')
  expect(splitOfPath(['images', 'valid', 'a.jpg'])).toBe('validation')
  expect(splitOfPath(['images', 'Test', 'a.jpg'])).toBe('test')
  expect(splitOfPath(['train', 'sub', 'a.jpg'])).toBe('train')
  expect(splitOfPath(['images', 'a.jpg'])).toBeNull()
  expect(splitOfPath(['a.jpg'])).toBeNull()
  // The file's own name is not a folder: a file called train.jpg is not in the train split.
  expect(splitOfPath(['images', 'train.jpg'])).toBeNull()
})

test('class names from a data.yaml: inline list, block list and index map', () => {
  expect(parseYamlNames('path: .\ntrain: images/train\nnames: [\'cat\', "dog", bird]\nnc: 3')).toEqual([
    'cat',
    'dog',
    'bird',
  ])
  expect(parseYamlNames('train: x\nnames:\n  - cat\n  - "big dog"  # a comment\n  - bird\nnc: 3')).toEqual([
    'cat',
    'big dog',
    'bird',
  ])
  expect(parseYamlNames('names:\n  0: cat\n  1: dog\n  2: bird\n')).toEqual(['cat', 'dog', 'bird'])
  expect(parseYamlNames('names:\n  1: dog\n  0: cat\n')).toEqual(['cat', 'dog']) // order of the lines is irrelevant
})

test('class names keep punctuation and commas inside quotes', () => {
  expect(parseYamlNames("names: ['red, square', 'green: odd', 'it''s']")).toEqual(['red, square', 'green: odd', "it's"])
  expect(parseYamlNames('names:\n  - "a # not a comment"\n')).toEqual(['a # not a comment'])
})

test('an unreadable names list is null rather than a guess', () => {
  expect(parseYamlNames('train: x\nval: y')).toBeNull() // no names key
  expect(parseYamlNames('names:\n  0: cat\n  2: bird\n')).toBeNull() // index 1 is missing
  expect(parseYamlNames('names:\ntrain: x')).toBeNull() // names is empty
})

test('a names file is yaml by extension and one class per line otherwise', () => {
  expect(parseNamesFile('data.yaml', 'names: [a, b]')).toEqual(['a', 'b'])
  expect(parseNamesFile('classes.txt', 'cat\r\n dog \n\nbird\n')).toEqual(['cat', 'dog', 'bird'])
  expect(parseNamesFile('classes.txt', '  \n')).toBeNull()
})

test('a label line is converted from centre and size to corner and size', () => {
  const { boxes, skipped } = parseLabelFile('1 0.5 0.5 0.2 0.4', 3)
  expect(skipped).toBe(0)
  expect(boxes).toHaveLength(1)
  expect(boxes[0].classIndex).toBe(1)
  expect(boxes[0].box.x).toBeCloseTo(0.4, 10)
  expect(boxes[0].box.y).toBeCloseTo(0.3, 10)
  expect(boxes[0].box.w).toBeCloseTo(0.2, 10)
  expect(boxes[0].box.h).toBeCloseTo(0.4, 10)
})

test('a box that pokes past the edge by rounding is clipped to the image', () => {
  const { boxes } = parseLabelFile('0 0.9999 0.5 0.0004 0.2\n0 0.5 0.0 0.2 0.2', 1)
  expect(boxes).toHaveLength(2)
  expect(boxes[0].box.x + boxes[0].box.w).toBeLessThanOrEqual(1)
  expect(boxes[1].box.y).toBe(0) // its top half was off the image: clipped, not rejected
  expect(boxes[1].box.h).toBeCloseTo(0.1, 10)
})

test('lines that cannot be a box are skipped and counted, never turned into something else', () => {
  const text = [
    '0 0.5 0.5 0.2 0.2', // fine
    '', // blank lines are not counted
    '0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4', // a segmentation polygon
    '5 0.5 0.5 0.2 0.2', // class 5 of 3
    '-1 0.5 0.5 0.2 0.2',
    '0.5 0.5 0.5 0.2 0.2', // a fractional class index
    'x 0.5 0.5 0.2 0.2',
    '0 0.5 0.5 0 0.2', // no width
    '0 1.5 0.5 0.1 0.1', // entirely off the image
  ].join('\n')
  const { boxes, skipped } = parseLabelFile(text, 3)
  expect(boxes).toHaveLength(1)
  expect(skipped).toBe(7)
})

test('an empty label file means no boxes, and every box line is read in order', () => {
  expect(parseLabelFile('', 2)).toEqual({ boxes: [], skipped: 0 })
  expect(parseLabelFile('\n  \n', 2)).toEqual({ boxes: [], skipped: 0 })
  expect(parseLabelFile('0 0.2 0.2 0.1 0.1\n1 0.7 0.7 0.1 0.1\n', 2).boxes.map((b) => b.classIndex)).toEqual([0, 1])
})

test('planning pairs images with labels, keeps splits, and counts what is missing or left over', () => {
  const images = [
    entry('ds', 'images', 'train', 'a.jpg'),
    entry('ds', 'images', 'val', 'b.jpg'),
    entry('ds', 'images', 'test', 'c.png'), // no label file
    entry('ds', 'images', 'train', 'd.jpg'), // an empty label file: background
  ]
  const labels = new Map([
    [pairKey(['ds', 'labels', 'train', 'a.txt']), '0 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.2 0.2'],
    [pairKey(['ds', 'labels', 'val', 'b.txt']), '1 0.5 0.5 0.2 0.2'],
    [pairKey(['ds', 'labels', 'train', 'd.txt']), ''],
    [pairKey(['ds', 'labels', 'train', 'orphan.txt']), '0 0.5 0.5 0.2 0.2'], // no image for it
  ])

  const plan = planImport(images, labels, 2)

  expect(plan.items.map((i) => [i.file.name, i.split, i.boxes?.length ?? null])).toEqual([
    ['a.jpg', 'train', 1], // its second line named a class that does not exist
    ['b.jpg', 'validation', 1],
    ['c.png', 'test', null], // unlabelled
    ['d.jpg', 'train', 0], // labelled, with nothing in it
  ])
  expect(plan.withoutLabels).toBe(1)
  expect(plan.skippedLines).toBe(1)
  expect(plan.orphanLabels).toBe(1)
})

test('with missingAsEmpty an image without a label file is read as empty instead of unlabelled', () => {
  const images = [entry('images', 'a.jpg')]
  expect(planImport(images, new Map(), 1).items[0].boxes).toBeNull()
  expect(planImport(images, new Map(), 1, { missingAsEmpty: true }).items[0].boxes).toEqual([])
})

test('upload boxes follow the files, with null for unlabelled and [] for empty', () => {
  const items = [
    { file: file('a.jpg'), split: null, boxes: [{ classIndex: 1, box: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 } }] },
    { file: file('b.jpg'), split: null, boxes: [] },
    { file: file('c.jpg'), split: null, boxes: null },
  ]
  expect(uploadBoxes(items, ['id-cat', 'id-dog'])).toEqual([
    [{ classId: 'id-dog', x: 0.1, y: 0.2, w: 0.3, h: 0.4 }],
    [],
    null,
  ])
})

test('chunks hold one split each, at most 50 files and 64 MB, in order', () => {
  const item = (name: string, split: 'train' | 'validation' | 'test' | null, size = 10) => ({
    file: file(name, size),
    split,
    boxes: null,
  })
  const many = Array.from({ length: 120 }, (_, i) => item(`t${i}`, 'train'))
  const chunks = chunkBySplit([...many, item('v', 'validation'), item('d', null)], 'test')

  expect(chunks.map((c) => [c.split, c.items.length])).toEqual([
    ['train', 50],
    ['train', 50],
    ['train', 20],
    ['validation', 1],
    ['test', 1], // no split in its path: the default
  ])
  const big = chunkBySplit([item('a', 'train', 40 * 1024 * 1024), item('b', 'train', 40 * 1024 * 1024)], 'train')
  expect(big.map((c) => c.items.length)).toEqual([1, 1]) // together they would pass 64 MB
})

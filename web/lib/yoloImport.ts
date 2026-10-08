/**
 * Reading a YOLO-format dataset: images, one `.txt` label file per image, and the class names.
 *
 * ```
 * images/train/a.jpg        labels/train/a.txt     "<class index> <cx> <cy> <w> <h>" per box, all 0..1, centre-based
 * data.yaml  (or classes.txt)                       the class names, in index order
 * ```
 *
 * Everything here is pure (no files are read, nothing is sent), so the format's edge cases can be tested directly. The
 * importer component does the reading and uploading around it.
 */

import type { Box } from './boxes'

export type ImportSplit = 'train' | 'validation' | 'test'

/** An extracted file: its folder path (ending in the file name) and the file itself. */
export interface YoloEntry {
  path: string[]
  file: File
}

/** One labelled box, with the class as the INDEX the label file used. */
export interface YoloBox {
  classIndex: number
  box: Box
}

/** The server takes only these as images (see the upload route's accepted types). */
const IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png'])
const NAMES_FILES = new Set(['data.yaml', 'data.yml', 'classes.txt', 'obj.names', 'names.txt'])
const SPLIT_FOLDERS: Record<string, ImportSplit> = {
  train: 'train',
  training: 'train',
  val: 'validation',
  valid: 'validation',
  validation: 'validation',
  test: 'test',
}

const extensionOf = (name: string) => {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot + 1).toLowerCase()
}

const stemOf = (name: string) => {
  const dot = name.lastIndexOf('.')
  return dot <= 0 ? name : name.slice(0, dot)
}

export const isImageName = (name: string) => IMAGE_EXTENSIONS.has(extensionOf(name))
export const isNamesFileName = (name: string) => NAMES_FILES.has(name.toLowerCase())
/** A label file: a `.txt` that is not the class list. */
export const isLabelName = (name: string) => extensionOf(name) === 'txt' && !isNamesFileName(name)

/**
 * What an image and its label file have in common, so they can be paired. Ultralytics finds labels by swapping the
 * last `images` folder for `labels`; a dataset with both side by side in one folder pairs by path alone.
 */
export const pairKey = (path: string[]): string => {
  const dirs = path.slice(0, -1)
  const marker = dirs.findLastIndex((d) => d.toLowerCase() === 'images' || d.toLowerCase() === 'labels')
  const normalised = dirs.map((d, i) => (i === marker ? '*' : d))
  return [...normalised, stemOf(path[path.length - 1])].join('/')
}

/** The split a file's folders name (`images/train/...`), nearest folder first, or null when they name none. */
export const splitOfPath = (path: string[]): ImportSplit | null => {
  for (let i = path.length - 2; i >= 0; i--) {
    const split = SPLIT_FOLDERS[path[i].toLowerCase()]
    if (split) return split
  }
  return null
}

const unquote = (value: string) => {
  const v = value.trim()
  if (v.length < 2 || (v[0] !== '"' && v[0] !== "'") || v[v.length - 1] !== v[0]) return v
  const inner = v.slice(1, -1)
  // In YAML a doubled single quote inside a single-quoted string is one apostrophe.
  return v[0] === "'" ? inner.replaceAll("''", "'") : inner
}

/** Drops a trailing ` # comment` that is not inside quotes. */
const stripComment = (line: string) => {
  let quote: string | null = null
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (quote) {
      if (ch === quote) quote = null
    } else if (ch === '"' || ch === "'") {
      quote = ch
    } else if (ch === '#' && (i === 0 || /\s/.test(line[i - 1]))) {
      return line.slice(0, i)
    }
  }
  return line
}

/** Splits `a, 'b, c', d` on the commas that are not inside quotes. */
const splitInlineList = (inner: string): string[] => {
  const parts: string[] = []
  let current = ''
  let quote: string | null = null
  for (const ch of inner) {
    if (quote) {
      current += ch
      if (ch === quote) quote = null
    } else if (ch === '"' || ch === "'") {
      quote = ch
      current += ch
    } else if (ch === ',') {
      parts.push(current)
      current = ''
    } else {
      current += ch
    }
  }
  if (current.trim()) parts.push(current)
  return parts.map(unquote).filter((p) => p !== '')
}

/**
 * The class names in a `data.yaml`: `names: [a, b]`, a block list of `- a` lines, or an index map (`0: a`).
 * Not a YAML parser: just those three shapes, which is what YOLO tools write. Null when none is found, or when an
 * index map has a gap (the names could not be put in index order).
 */
export const parseYamlNames = (text: string): string[] | null => {
  const lines = text.split(/\r?\n/).map(stripComment)
  const start = lines.findIndex((l) => /^names\s*:/.test(l))
  if (start === -1) return null

  const rest = lines[start].replace(/^names\s*:/, '').trim()
  if (rest.startsWith('[')) {
    const close = rest.lastIndexOf(']')
    return splitInlineList(rest.slice(1, close === -1 ? undefined : close))
  }

  const listed: string[] = []
  const indexed = new Map<number, string>()
  for (const line of lines.slice(start + 1)) {
    if (!line.trim()) continue
    if (!/^\s/.test(line) && !line.trimStart().startsWith('-')) break // the next top-level key
    const item = /^\s*-\s*(.*)$/.exec(line)
    if (item) {
      listed.push(unquote(item[1]))
      continue
    }
    const mapped = /^\s*(\d+)\s*:\s*(.*)$/.exec(line)
    if (mapped) indexed.set(Number(mapped[1]), unquote(mapped[2]))
  }
  if (listed.length > 0) return listed
  if (indexed.size === 0) return null
  const names: string[] = []
  for (let i = 0; i < indexed.size; i++) {
    const name = indexed.get(i)
    if (name === undefined) return null
    names.push(name)
  }
  return names
}

/** The class names a names file holds: a `data.yaml`'s `names`, or one class per line of a `classes.txt`. */
export const parseNamesFile = (fileName: string, text: string): string[] | null => {
  if (/\.ya?ml$/i.test(fileName)) return parseYamlNames(text)
  const names = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
  return names.length > 0 ? names : null
}

/**
 * The boxes in one label file. A line is `class cx cy w h`; its box is converted from centre-based to the top-left
 * convention the server stores, and clipped to the image, since exporters round past the edge by a hair. A line that
 * cannot be a box (polygon points, a class index the names do not cover, nothing left after clipping) is skipped
 * and counted, so the caller can say how much was left out.
 */
export const parseLabelFile = (text: string, classCount: number): { boxes: YoloBox[]; skipped: number } => {
  const boxes: YoloBox[] = []
  let skipped = 0
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line) continue
    const parts = line.split(/\s+/)
    const numbers = parts.map(Number)
    const [classIndex, cx, cy, w, h] = numbers
    if (parts.length !== 5 || !numbers.every(Number.isFinite) || !Number.isInteger(classIndex)) {
      skipped++
      continue
    }
    if (classIndex < 0 || classIndex >= classCount) {
      skipped++
      continue
    }
    const x0 = Math.max(0, cx - w / 2)
    const y0 = Math.max(0, cy - h / 2)
    const x1 = Math.min(1, cx + w / 2)
    const y1 = Math.min(1, cy + h / 2)
    if (x1 - x0 <= 0 || y1 - y0 <= 0) {
      skipped++
      continue
    }
    boxes.push({ classIndex, box: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } })
  }
  return { boxes, skipped }
}

/** An image of the dataset, with its split and what its label file said. */
export interface ImportItem {
  file: File
  /** The split its folders name, else null (the caller supplies a default). */
  split: ImportSplit | null
  /** null: not labelled. [] : labelled, with nothing in it. */
  boxes: YoloBox[] | null
}

export interface ImportPlan {
  items: ImportItem[]
  /** Images that have no label file (unlabelled, or empty if `missingAsEmpty`). */
  withoutLabels: number
  /** Label lines that could not be turned into a box. */
  skippedLines: number
  /** Label files with no image to go with them. */
  orphanLabels: number
}

/**
 * Pairs images with label files and reads the labels.
 *
 * `labelTexts` maps `pairKey` to the contents of a label file. An image with no label file is unlabelled by default;
 * `missingAsEmpty` reads it as "nothing in this image" instead, which is how some datasets mark background images.
 */
export const planImport = (
  images: YoloEntry[],
  labelTexts: Map<string, string>,
  classCount: number,
  options: { missingAsEmpty?: boolean } = {},
): ImportPlan => {
  let withoutLabels = 0
  let skippedLines = 0
  const used = new Set<string>()
  const items = images.map((image): ImportItem => {
    const key = pairKey(image.path)
    const text = labelTexts.get(key)
    const split = splitOfPath(image.path)
    if (text === undefined) {
      withoutLabels++
      return { file: image.file, split, boxes: options.missingAsEmpty ? [] : null }
    }
    used.add(key)
    const parsed = parseLabelFile(text, classCount)
    skippedLines += parsed.skipped
    return { file: image.file, split, boxes: parsed.boxes }
  })
  return { items, withoutLabels, skippedLines, orphanLabels: [...labelTexts.keys()].filter((k) => !used.has(k)).length }
}

/** The `boxes` field the upload route takes for a chunk of files: one entry per file, in order. */
export const uploadBoxes = (items: ImportItem[], classIdByIndex: string[]) =>
  items.map((item) =>
    item.boxes === null
      ? null
      : item.boxes.map(({ classIndex, box }) => ({ classId: classIdByIndex[classIndex], ...box })),
  )

const MAX_CHUNK_FILES = 50
const MAX_CHUNK_BYTES = 64 * 1024 * 1024

/**
 * Groups items into upload requests: one split per request (the route takes one), at most 50 files and 64 MB each,
 * so a request stays small enough to retry.
 */
export const chunkBySplit = (
  items: ImportItem[],
  defaultSplit: ImportSplit,
): { split: ImportSplit; items: ImportItem[] }[] => {
  const chunks: { split: ImportSplit; items: ImportItem[]; bytes: number }[] = []
  const open = new Map<ImportSplit, (typeof chunks)[number]>()
  for (const item of items) {
    const split = item.split ?? defaultSplit
    let chunk = open.get(split)
    if (!chunk || chunk.items.length >= MAX_CHUNK_FILES || chunk.bytes + item.file.size > MAX_CHUNK_BYTES) {
      chunk = { split, items: [], bytes: 0 }
      chunks.push(chunk)
      open.set(split, chunk)
    }
    chunk.items.push(item)
    chunk.bytes += item.file.size
  }
  return chunks.map(({ split, items }) => ({ split, items }))
}

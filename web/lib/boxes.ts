/**
 * Bounding boxes, as the server stores them (see `server/omnitrain/services/boxes.py`).
 *
 * One annotation per box: `labelStructured` is `{x, y, w, h}`, the top-left corner and the size, each a fraction
 * (0..1) of the whole image as it is displayed. An image with no objects carries one marker annotation,
 * `{background: true}`, with no class. Everything here is pure, so the drawing rules can be tested without a DOM.
 */

import type { Annotation } from '@public/store/types'

export interface Box {
  x: number
  y: number
  w: number
  h: number
}

/** A point in image fractions, 0..1 on both axes. */
export interface Point {
  x: number
  y: number
}

/** A box on an item, with the annotation that holds it. */
export interface ItemBox {
  /** The annotation id. */
  id: string
  classId: string | null
  box: Box
}

/** Dragging less than this fraction of the image (in either direction) is a click, not a box. */
export const MIN_BOX_SIZE = 0.01

const BOX_KEYS = ['x', 'y', 'w', 'h'] as const

export const clamp01 = (n: number) => Math.min(1, Math.max(0, n))

const isFraction = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** The box in an annotation's `labelStructured`, or null if it is not one (a marker, or a malformed value). */
export const readBox = (labelStructured: unknown): Box | null => {
  if (!labelStructured || typeof labelStructured !== 'object') return null
  const value = labelStructured as Record<string, unknown>
  if (!BOX_KEYS.every((key) => isFraction(value[key]))) return null
  const { x, y, w, h } = value as unknown as Box
  return w > 0 && h > 0 ? { x, y, w, h } : null
}

export const isBackgroundMarker = (labelStructured: unknown): boolean =>
  !!labelStructured &&
  typeof labelStructured === 'object' &&
  (labelStructured as { background?: unknown }).background === true

const ofType = (annotations: Annotation[] | undefined) =>
  (annotations ?? []).filter((a) => a.annotationType === 'bounding_box')

/** The item's boxes, in the order they were drawn. */
export const boxesOf = (annotations: Annotation[] | undefined): ItemBox[] =>
  ofType(annotations).flatMap((a) => {
    const box = readBox(a.labelStructured)
    return box ? [{ id: a.id, classId: a.classId ?? null, box }] : []
  })

/** The "no objects" marker annotation, when the item carries one. */
export const backgroundMarker = (annotations: Annotation[] | undefined) =>
  ofType(annotations).find((a) => isBackgroundMarker(a.labelStructured))

/** How an item reads in a list: how many boxes, "No objects", or null while nobody has labelled it. */
export const boxesSummary = (annotations: Annotation[] | undefined): string | null => {
  const count = boxesOf(annotations).length
  if (count > 0) return `${count} box${count === 1 ? '' : 'es'}`
  return backgroundMarker(annotations) ? 'No objects' : null
}

/** Where a pointer is on the image, as fractions and clamped to it, so dragging past an edge stops at the edge. */
export const pointOnImage = (
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
): Point => ({
  x: rect.width > 0 ? clamp01((clientX - rect.left) / rect.width) : 0,
  y: rect.height > 0 ? clamp01((clientY - rect.top) / rect.height) : 0,
})

/** The box spanned by a drag between two points (in any direction), or null if it is too small to be deliberate. */
export const boxFromDrag = (a: Point, b: Point): Box | null => {
  const x = Math.min(a.x, b.x)
  const y = Math.min(a.y, b.y)
  const w = Math.abs(a.x - b.x)
  const h = Math.abs(a.y - b.y)
  return w < MIN_BOX_SIZE || h < MIN_BOX_SIZE ? null : { x, y, w, h }
}

/** The box moved by (dx, dy) fractions, held inside the image. */
export const nudgeBox = (box: Box, dx: number, dy: number): Box => ({
  ...box,
  x: Math.min(1 - box.w, Math.max(0, box.x + dx)),
  y: Math.min(1 - box.h, Math.max(0, box.y + dy)),
})

/** The box under a point. Where boxes overlap the smallest wins, so a small box inside a big one stays selectable. */
export const hitTest = (boxes: ItemBox[], point: Point): ItemBox | null => {
  let best: ItemBox | null = null
  for (const candidate of boxes) {
    const { x, y, w, h } = candidate.box
    if (point.x < x || point.x > x + w || point.y < y || point.y > y + h) continue
    if (best === null || w * h < best.box.w * best.box.h) best = candidate
  }
  return best
}

/** CSS percentages for positioning an HTML element over the image, from a box. */
export const boxStyle = (box: Box) => ({
  left: `${box.x * 100}%`,
  top: `${box.y * 100}%`,
  width: `${box.w * 100}%`,
  height: `${box.h * 100}%`,
})

/** A detection as the server returns it: a box plus what it is and how sure the model is. */
export interface Detection {
  label: string
  confidence: number
  box: Box
}

/** The `detections` inference output (server `DetectionsOutput`). */
export interface DetectionsOutput {
  kind: 'detections'
  feature: string
  imageWidth: number
  imageHeight: number
  detections: Detection[]
}

const FALLBACK_COLORS = ['#e03131', '#2f9e44', '#1971c2', '#f08c00', '#9c36b5', '#0c8599', '#e8590c', '#6741d9']
const NO_CLASS_COLOR = '#868e96'

/**
 * The colour a class's boxes are drawn in: the class's own colour when it has one, else a colour that depends
 * only on the class id, so the same class looks the same wherever it appears.
 */
export const classColor = (classId: string | null | undefined, ownColors?: Map<string, string>): string => {
  if (!classId) return NO_CLASS_COLOR
  const own = ownColors?.get(classId)
  if (own) return own
  let hash = 0
  for (const ch of classId) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0
  return FALLBACK_COLORS[hash % FALLBACK_COLORS.length]
}

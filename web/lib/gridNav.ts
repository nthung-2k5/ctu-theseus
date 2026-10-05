/**
 * Arrow-key navigation over a grid of cards, decided by where the cards actually are on screen rather than
 * by an index and a column count: that makes it right for a CSS grid, for `VirtualGrid` (which only has some
 * rows mounted) and at any breakpoint. Pure, so it can be unit-tested without a DOM.
 */

export type Direction = 'left' | 'right' | 'up' | 'down'

export interface Box {
  left: number
  top: number
  width: number
  height: number
}

const centerX = (b: Box) => b.left + b.width / 2
const centerY = (b: Box) => b.top + b.height / 2

/**
 * The index of the card to move to from `boxes[from]`, or null at an edge (it never wraps). Left/right stay
 * within the current row and take the nearest card; up/down take the nearest row in that direction and, in it,
 * the card closest in x.
 */
export function nextInDirection(boxes: readonly Box[], from: number, direction: Direction): number | null {
  const current = boxes[from]
  if (!current) return null
  const cx = centerX(current)
  const cy = centerY(current)
  const half = current.height / 2

  let best: number | null = null
  let bestScore = Number.POSITIVE_INFINITY

  if (direction === 'left' || direction === 'right') {
    boxes.forEach((box, i) => {
      if (i === from || Math.abs(centerY(box) - cy) >= half) return // another row
      const dx = centerX(box) - cx
      if (direction === 'right' ? dx <= 0 : dx >= 0) return
      if (Math.abs(dx) < bestScore) {
        best = i
        bestScore = Math.abs(dx)
      }
    })
    return best
  }

  // Up/down: find the nearest row in that direction, then the closest column within it.
  let nearestRow = Number.POSITIVE_INFINITY
  for (const [i, box] of boxes.entries()) {
    if (i === from) continue
    const dy = centerY(box) - cy
    if (direction === 'down' ? dy <= half : dy >= -half) continue // same row, or the wrong way
    nearestRow = Math.min(nearestRow, Math.abs(dy))
  }
  if (nearestRow === Number.POSITIVE_INFINITY) return null
  boxes.forEach((box, i) => {
    if (i === from) return
    const dy = centerY(box) - cy
    if (direction === 'down' ? dy <= half : dy >= -half) return
    if (Math.abs(Math.abs(dy) - nearestRow) >= half) return // a farther row
    const dx = Math.abs(centerX(box) - cx)
    if (dx < bestScore) {
      best = i
      bestScore = dx
    }
  })
  return best
}

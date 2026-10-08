import { type Box, boxFromDrag, boxStyle, hitTest, type ItemBox, type Point, pointOnImage } from '@public/lib/boxes'
import { type KeyboardEvent, type PointerEvent, useRef, useState } from 'react'

/** One box to draw: its place on the image and how it should look. */
export interface CanvasBox {
  id: string
  box: Box
  color: string
  /** Shown in a small tag on the box's top-left corner. */
  label?: string
  selected?: boolean
}

/** How far an arrow key moves the selected box, as a fraction of the image; with Shift it moves faster. */
const NUDGE = 0.005
const NUDGE_FAST = 0.02

/** `#rrggbb` with an alpha, for a see-through fill. Anything that is not a 6-digit hex gets no fill. */
const withAlpha = (color: string, alpha: number) =>
  /^#[0-9a-f]{6}$/i.test(color)
    ? `${color}${Math.round(alpha * 255)
        .toString(16)
        .padStart(2, '0')}`
    : 'transparent'

/**
 * An image with boxes drawn over it, in image fractions so they stay on the right pixels at any size.
 *
 * Read-only unless `onDraw` is given. Then dragging draws a new box, clicking selects the box under the pointer
 * (or nothing), and, with the image focused, Delete removes the selected box, the arrow keys move it and Escape
 * lets go of it. The overlay is never itself a target: every pointer event lands on the image's wrapper, which
 * works out what was meant, so a box can be drawn right over another one.
 */
export const BoxCanvas = ({
  src,
  alt,
  boxes,
  drawColor = '#fa5252',
  onDraw,
  onSelect,
  onNudge,
  onDelete,
}: {
  src?: string | null
  alt: string
  boxes: CanvasBox[]
  /** The colour of the box being dragged out. */
  drawColor?: string
  onDraw?: (box: Box) => void
  onSelect?: (id: string | null) => void
  onNudge?: (id: string, dx: number, dy: number) => void
  onDelete?: (id: string) => void
}) => {
  const wrapper = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState<{ start: Point; current: Point } | null>(null)
  const interactive = onDraw !== undefined
  const selected = boxes.find((b) => b.selected)

  const pointer = (e: PointerEvent) => {
    const rect = wrapper.current?.getBoundingClientRect()
    return rect ? pointOnImage(e.clientX, e.clientY, rect) : { x: 0, y: 0 }
  }

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!interactive || e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    e.currentTarget.focus()
    const start = pointer(e)
    setDrag({ start, current: start })
  }

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (drag) setDrag({ ...drag, current: pointer(e) })
  }

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    if (!drag) return
    const end = pointer(e)
    const drawn = boxFromDrag(drag.start, end)
    setDrag(null)
    if (drawn) {
      onDraw?.(drawn)
      return
    }
    // Too small to be a box: it was a click.
    const items: ItemBox[] = boxes.map((b) => ({ id: b.id, classId: null, box: b.box }))
    onSelect?.(hitTest(items, end)?.id ?? null)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!interactive) return
    // The drawer steps to the next item on the arrow keys; while a box is selected they belong to the box.
    const claim = () => {
      e.preventDefault()
      e.stopPropagation()
    }
    if (e.key === 'Escape' && selected) {
      claim()
      onSelect?.(null)
      return
    }
    if (!selected) return
    if (e.key === 'Delete' || e.key === 'Backspace') {
      claim()
      onDelete?.(selected.id)
      return
    }
    const step = e.shiftKey ? NUDGE_FAST : NUDGE
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    }
    const move = moves[e.key]
    if (move) {
      claim()
      onNudge?.(selected.id, move[0], move[1])
    }
  }

  const dragged = drag ? boxFromDrag(drag.start, drag.current) : null
  // An element with handlers needs a role and a name; a plain image does not.
  const interactiveProps = interactive
    ? {
        role: 'application' as const,
        'aria-label': 'Bounding box editor. Drag to draw a box, click a box to select it.',
        tabIndex: 0,
        onPointerDown,
        onPointerMove,
        onPointerUp,
        onPointerCancel: () => setDrag(null),
        onKeyDown,
      }
    : { role: 'img' as const, 'aria-label': alt }

  return (
    <div
      ref={wrapper}
      {...interactiveProps}
      style={{
        position: 'relative',
        lineHeight: 0,
        borderRadius: 8,
        overflow: 'hidden',
        cursor: interactive ? 'crosshair' : undefined,
        touchAction: interactive ? 'none' : undefined,
        userSelect: 'none',
        outline: 'none',
      }}
    >
      {/* The wrapper takes the image's own shape, so a fraction of the wrapper is a fraction of the image. */}
      <img
        src={src ?? undefined}
        alt={alt}
        draggable={false}
        style={{ display: 'block', width: '100%', height: 'auto' }}
      />
      {boxes.map((b) => (
        <div
          key={b.id}
          style={{
            position: 'absolute',
            ...boxStyle(b.box),
            boxSizing: 'border-box',
            border: `${b.selected ? 3 : 2}px solid ${b.color}`,
            background: withAlpha(b.color, b.selected ? 0.28 : 0.14),
            boxShadow: b.selected ? '0 0 0 1px #fff' : undefined,
            pointerEvents: 'none',
          }}
        >
          {b.label && (
            <span
              style={{
                position: 'absolute',
                // Inside the box, so a box touching the top edge of the image still shows its tag.
                left: 0,
                top: 0,
                maxWidth: '100%',
                padding: '1px 4px',
                font: '600 10px/1.3 system-ui, sans-serif',
                color: '#fff',
                background: b.color,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {b.label}
            </span>
          )}
        </div>
      ))}
      {dragged && (
        <div
          style={{
            position: 'absolute',
            ...boxStyle(dragged),
            boxSizing: 'border-box',
            border: `2px dashed ${drawColor}`,
            background: withAlpha(drawColor, 0.12),
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  )
}

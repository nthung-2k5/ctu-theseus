/**
 * A small preview of one staged file.
 *
 * A dataset folder can hold thousands of full-size photos, so a preview is never the photo itself: it is decoded at a
 * small size (`createImageBitmap` with a resize hint, which lets the browser skip most of the work for a JPEG), only
 * once its tile is on screen, and only a few at a time. Anything that is not an image, or cannot be decoded, shows an
 * icon for its kind instead.
 */

import { FileAudioIcon, FileIcon, FileVideoIcon, ImageIcon } from '@phosphor-icons/react'
import { createLimiter } from '@public/lib/upload/lanes'
import type { ReviewRow } from '@public/lib/upload/review'
import { useEffect, useRef, useState } from 'react'

const DECODE_WIDTH = 192
const limit = createLimiter(3)

/** True once the element has been on screen (and stays true: a decoded tile is not thrown away on scroll). */
function useSeen(ref: React.RefObject<HTMLElement | null>): boolean {
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setSeen(true)
        observer.disconnect()
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref, seen])
  return seen
}

export function Thumb({
  file,
  media,
  size,
}: {
  /** The image to preview; leave out for a file that has none to show. */
  file?: File
  media?: ReviewRow['media']
  /** Pixels, or omit to fill the parent (which then sets the shape). */
  size?: number
}) {
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const seen = useSeen(box)
  const [drawn, setDrawn] = useState(false)

  useEffect(() => {
    if (!file || !seen) return
    const controller = new AbortController()
    setDrawn(false)
    limit(async () => {
      const bitmap = await createImageBitmap(file, { resizeWidth: DECODE_WIDTH, resizeQuality: 'low' })
      try {
        const target = canvas.current
        if (controller.signal.aborted || !target) return
        target.width = bitmap.width
        target.height = bitmap.height
        target.getContext('2d')?.drawImage(bitmap, 0, 0)
        setDrawn(true)
      } finally {
        bitmap.close()
      }
    }, controller.signal).catch(() => {
      // Aborted, or not a picture the browser can decode: the icon stays.
    })
    return () => controller.abort()
  }, [file, seen])

  const Icon =
    media === 'audio' ? FileAudioIcon : media === 'video' ? FileVideoIcon : media === 'image' ? ImageIcon : FileIcon
  return (
    <div
      ref={box}
      style={{
        width: size ?? '100%',
        height: size ?? '100%',
        borderRadius: 'var(--mantine-radius-sm)',
        background: 'var(--mantine-color-default-hover)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        flexShrink: 0,
        position: 'relative',
      }}
    >
      {!drawn && <Icon size={size ? Math.round(size / 2) : 22} color="var(--mantine-color-dimmed)" />}
      {/* biome-ignore lint/a11y/noAriaHiddenOnFocusable: a canvas without fallback content is not focusable; this one is a decorative preview */}
      <canvas
        ref={canvas}
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: drawn ? 'block' : 'none',
        }}
      />
    </div>
  )
}

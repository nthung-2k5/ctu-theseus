import { type CSSProperties, type ReactNode, useEffect, useRef, useState } from 'react'

const prefersReduced = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** True once the element has scrolled into view (fires once). Immediately true under reduced motion. */
export function useInView<T extends HTMLElement>(threshold = 0.2) {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(prefersReduced())
  useEffect(() => {
    if (seen || !ref.current) return
    const io = new IntersectionObserver(
      ([e]) => {
        if (e?.isIntersecting) {
          setSeen(true)
          io.disconnect()
        }
      },
      { threshold },
    )
    io.observe(ref.current)
    return () => io.disconnect()
  }, [seen, threshold])
  return [ref, seen] as const
}

/** Fades and lifts its children in when scrolled into view, with an optional stagger delay (ms). */
export function Reveal({
  children,
  delay = 0,
  className = '',
  style,
}: {
  children: ReactNode
  delay?: number
  className?: string
  style?: CSSProperties
}) {
  const [ref, seen] = useInView<HTMLDivElement>(0.15)
  return (
    <div
      ref={ref}
      className={`ot-reveal ${seen ? 'is-in' : ''} ${className}`}
      style={{ transitionDelay: `${delay}ms`, ...style }}
    >
      {children}
    </div>
  )
}

/** Counts up to `to` when scrolled into view (easeOutCubic). */
export function CountUp({
  to,
  decimals = 0,
  suffix = '',
  duration = 1600,
}: {
  to: number
  decimals?: number
  suffix?: string
  duration?: number
}) {
  const [ref, seen] = useInView<HTMLSpanElement>(0.4)
  const [v, setV] = useState(prefersReduced() ? to : 0)
  useEffect(() => {
    if (!seen || prefersReduced()) {
      if (seen) setV(to)
      return
    }
    let raf = 0
    const t0 = performance.now()
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / duration)
      setV(to * (1 - (1 - k) ** 3))
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [seen, to, duration])
  return (
    <span ref={ref} className="ot-tnum">
      {v.toFixed(decimals)}
      {suffix}
    </span>
  )
}

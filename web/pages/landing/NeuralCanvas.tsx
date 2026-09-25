import { useEffect, useRef } from 'react'

interface Node {
  x: number
  y: number
  layer: number
  glow: number
  phase: number
  out: number[]
}
interface Edge {
  from: number
  to: number
}
interface Signal {
  e: number
  t: number
  speed: number
}

interface Props {
  /** Node density multiplier. */
  density?: number
  /** Nodes lean toward and light up near the pointer. */
  interactive?: boolean
  className?: string
}

// Small deterministic PRNG so the network layout is stable across resizes.
function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Animated feed-forward network: layers of nodes, faint edges, and glowing "activations" that
 * travel layer to layer. Pauses when off-screen or when the tab is hidden, and renders one
 * static frame for users who prefer reduced motion.
 */
export function NeuralCanvas({ density = 1, interactive = true, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let w = 0,
      h = 0,
      dpr = 1
    let nodes: Node[] = []
    let edges: Edge[] = []
    let signals: Signal[] = []
    let layers = 5
    let raf = 0
    let running = false
    let visible = true
    const mouse = { x: -9999, y: -9999, active: false }
    const parallax = { x: 0, y: 0 }

    const layout = () => {
      const r = rng(7)
      layers = w < 640 ? 4 : w < 1100 ? 5 : 7
      const perLayer = Math.max(4, Math.min(11, Math.round((h / 78) * density)))
      nodes = []
      for (let l = 0; l < layers; l++) {
        const count = l === 0 || l === layers - 1 ? Math.max(3, Math.round(perLayer * 0.6)) : perLayer
        for (let i = 0; i < count; i++) {
          nodes.push({
            x: (0.06 + (0.88 * l) / (layers - 1)) * w,
            y: ((i + 0.5 + (r() - 0.5) * 0.5) / count) * h,
            layer: l,
            glow: 0,
            phase: r() * Math.PI * 2,
            out: [],
          })
        }
      }
      edges = []
      const byLayer: number[][] = Array.from({ length: layers }, () => [])
      nodes.forEach((n, i) => {
        byLayer[n.layer]!.push(i)
      })
      for (let l = 0; l < layers - 1; l++) {
        for (const from of byLayer[l]!) {
          const targets = byLayer[l + 1]!
          const k = Math.min(targets.length, 2 + Math.floor(r() * 2))
          const picked = new Set<number>()
          while (picked.size < k) picked.add(targets[Math.floor(r() * targets.length)]!)
          for (const to of picked) {
            nodes[from]!.out.push(edges.length)
            edges.push({ from, to })
          }
        }
      }
      signals = []
    }

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = canvas.clientWidth
      h = canvas.clientHeight
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      layout()
      if (!running) draw(0)
    }

    const pos = (n: Node, t: number) => {
      const drift = reduce ? 0 : Math.sin(t * 0.0006 + n.phase) * 5
      const depth = 0.4 + (n.layer / layers) * 0.6
      let x = n.x + parallax.x * depth * 16
      let y = n.y + drift + parallax.y * depth * 16
      if (interactive && mouse.active) {
        const dx = mouse.x - x,
          dy = mouse.y - y
        const d = Math.hypot(dx, dy)
        if (d < 150) {
          const f = (1 - d / 150) * 10
          x += (dx / d) * f
          y += (dy / d) * f
          n.glow = Math.max(n.glow, (1 - d / 150) * 0.9)
        }
      }
      return [x, y] as const
    }

    const spawn = (r: () => number) => {
      const starts = nodes.filter((n) => n.layer === 0 && n.out.length)
      const n = starts[Math.floor(r() * starts.length)]
      if (n) signals.push({ e: n.out[Math.floor(r() * n.out.length)]!, t: 0, speed: 0.35 + r() * 0.5 })
    }

    const rand = rng(99)
    let last = 0
    let spawnAcc = 0

    function draw(t: number) {
      const dt = Math.min(0.05, (t - last) / 1000 || 0)
      last = t
      ctx.clearRect(0, 0, w, h)
      const P = nodes.map((n) => pos(n, t))

      // edges
      ctx.lineWidth = 1
      ctx.strokeStyle = 'rgba(74,137,209,0.11)'
      ctx.beginPath()
      for (const e of edges) {
        const a = P[e.from]!,
          b = P[e.to]!
        ctx.moveTo(a[0], a[1])
        ctx.lineTo(b[0], b[1])
      }
      ctx.stroke()

      // signals (activations flowing through the network)
      if (!reduce) {
        spawnAcc += dt
        if (spawnAcc > 0.16 && signals.length < 26 * density + 6) {
          spawnAcc = 0
          spawn(rand)
        }
        for (let i = signals.length - 1; i >= 0; i--) {
          const s = signals[i]!
          s.t += dt * s.speed
          const e = edges[s.e]!
          const a = P[e.from]!,
            b = P[e.to]!
          const k = Math.min(1, s.t)
          const x = a[0] + (b[0] - a[0]) * k,
            y = a[1] + (b[1] - a[1]) * k
          const tail = Math.max(0, k - 0.22)
          const g = ctx.createLinearGradient(a[0] + (b[0] - a[0]) * tail, a[1] + (b[1] - a[1]) * tail, x, y)
          g.addColorStop(0, 'rgba(0,175,239,0)')
          g.addColorStop(1, 'rgba(0,175,239,0.95)')
          ctx.strokeStyle = g
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(a[0] + (b[0] - a[0]) * tail, a[1] + (b[1] - a[1]) * tail)
          ctx.lineTo(x, y)
          ctx.stroke()
          if (s.t >= 1) {
            const dest = nodes[e.to]!
            dest.glow = 1
            if (dest.out.length) {
              s.e = dest.out[Math.floor(rand() * dest.out.length)]!
              s.t = 0
            } else signals.splice(i, 1)
          }
        }
      }

      // nodes
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i]!
        const [x, y] = P[i]!
        n.glow = Math.max(0, n.glow - dt * 1.6)
        const r = 2.2 + n.glow * 3.2
        if (n.glow > 0.05) {
          const g = ctx.createRadialGradient(x, y, 0, x, y, r * 5)
          g.addColorStop(0, `rgba(0,175,239,${0.5 * n.glow})`)
          g.addColorStop(1, 'rgba(0,175,239,0)')
          ctx.fillStyle = g
          ctx.beginPath()
          ctx.arc(x, y, r * 5, 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.fillStyle = n.glow > 0.05 ? `rgba(160,225,250,${0.6 + 0.4 * n.glow})` : 'rgba(120,160,215,0.55)'
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    const loop = (t: number) => {
      if (!running) return
      // ease the parallax toward the pointer
      const tx = mouse.active ? mouse.x / w - 0.5 : 0,
        ty = mouse.active ? mouse.y / h - 0.5 : 0
      parallax.x += (tx - parallax.x) * 0.05
      parallax.y += (ty - parallax.y) * 0.05
      draw(t)
      raf = requestAnimationFrame(loop)
    }
    const start = () => {
      if (!running && !reduce && visible && !document.hidden) {
        running = true
        last = performance.now()
        raf = requestAnimationFrame(loop)
      }
    }
    const stop = () => {
      running = false
      cancelAnimationFrame(raf)
    }

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect()
      mouse.x = e.clientX - r.left
      mouse.y = e.clientY - r.top
      mouse.active = true
    }
    const onLeave = () => {
      mouse.active = false
    }
    const onVis = () => (document.hidden ? stop() : start())

    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    const io = new IntersectionObserver(([entry]) => {
      visible = !!entry?.isIntersecting
      visible ? start() : stop()
    })
    io.observe(canvas)
    if (interactive) {
      window.addEventListener('pointermove', onMove, { passive: true })
      document.addEventListener('pointerleave', onLeave)
    }
    document.addEventListener('visibilitychange', onVis)
    resize()
    start()
    if (reduce) draw(0)

    return () => {
      stop()
      ro.disconnect()
      io.disconnect()
      window.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerleave', onLeave)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [density, interactive])

  return (
    <canvas
      ref={ref}
      className={className}
      aria-hidden="true"
      tabIndex={-1}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    />
  )
}

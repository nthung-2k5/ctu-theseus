import { MONO_STACK } from '@public/theme'
import { useEffect, useRef, useState } from 'react'

const N = 160

/** Deterministic pseudo-noise so the static (reduced-motion) frame is stable. */
const noise = (i: number, k = 1) => (Math.sin(i * 12.9898 * k) * 43758.5453) % 1

function sample(step: number) {
  const prog = 1 - Math.exp(-step / 420)
  const loss = 2.25 * (1 - prog) + 0.17 + noise(step, 1) * 0.045 * (1 - prog * 0.5)
  const acc = Math.min(0.985, 0.1 + 0.87 * prog + noise(step, 2) * 0.012)
  return { loss, acc }
}

interface Line {
  id: number
  step: number
  loss: number
  acc: number
}

/**
 * A believable, *actually animated* training card: a canvas chart that draws itself at 20 points/s,
 * counters that tick, and a console that streams lines: the same experience as the real Live view.
 */
export function LiveCard() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [stats, setStats] = useState({ step: 0, loss: 2.25, acc: 0.1 })
  const [lines, setLines] = useState<Line[]>([])
  const state = useRef({ step: 0, loss: [] as number[], acc: [] as number[] })

  useEffect(() => {
    const cv = canvas.current!
    const ctx = cv.getContext('2d')!
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf = 0
    let acc = 0
    let last = performance.now()
    let lineId = 0
    let visible = true

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      cv.width = Math.round(cv.clientWidth * dpr)
      cv.height = Math.round(cv.clientHeight * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const push = () => {
      const s = state.current
      s.step += 1
      const p = sample(s.step)
      s.loss.push(p.loss)
      s.acc.push(p.acc)
      if (s.loss.length > N) {
        s.loss.shift()
        s.acc.shift()
      }
      return p
    }

    const line = (arr: number[], min: number, max: number, w: number, h: number, pad: number) => {
      ctx.beginPath()
      arr.forEach((v, i) => {
        const x = pad + (i / (N - 1)) * (w - pad * 2)
        const y = h - pad - ((v - min) / (max - min)) * (h - pad * 2)
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
      })
    }

    const draw = () => {
      const w = cv.clientWidth,
        h = cv.clientHeight,
        pad = 14
      ctx.clearRect(0, 0, w, h)
      ctx.strokeStyle = 'rgba(148,163,184,0.10)'
      ctx.lineWidth = 1
      for (let i = 1; i < 4; i++) {
        ctx.beginPath()
        ctx.moveTo(0, (h / 4) * i)
        ctx.lineTo(w, (h / 4) * i)
        ctx.stroke()
      }
      const s = state.current
      if (s.loss.length < 2) return

      // loss area + line
      const g = ctx.createLinearGradient(0, 0, 0, h)
      g.addColorStop(0, 'rgba(0,175,239,0.28)')
      g.addColorStop(1, 'rgba(0,175,239,0)')
      line(s.loss, 0, 2.4, w, h, pad)
      ctx.lineTo(pad + ((s.loss.length - 1) / (N - 1)) * (w - pad * 2), h)
      ctx.lineTo(pad, h)
      ctx.closePath()
      ctx.fillStyle = g
      ctx.fill()
      line(s.loss, 0, 2.4, w, h, pad)
      ctx.strokeStyle = '#00afef'
      ctx.lineWidth = 2
      ctx.lineJoin = 'round'
      ctx.stroke()

      // accuracy
      line(s.acc, 0, 1, w, h, pad)
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 1.6
      ctx.setLineDash([5, 4])
      ctx.stroke()
      ctx.setLineDash([])

      // glowing head
      const i = s.loss.length - 1
      const hx = pad + (i / (N - 1)) * (w - pad * 2)
      const hy = h - pad - (s.loss[i]! / 2.4) * (h - pad * 2)
      const rg = ctx.createRadialGradient(hx, hy, 0, hx, hy, 16)
      rg.addColorStop(0, 'rgba(0,175,239,0.7)')
      rg.addColorStop(1, 'rgba(0,175,239,0)')
      ctx.fillStyle = rg
      ctx.beginPath()
      ctx.arc(hx, hy, 16, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#e0f7ff'
      ctx.beginPath()
      ctx.arc(hx, hy, 3, 0, Math.PI * 2)
      ctx.fill()
    }

    resize()
    const ro = new ResizeObserver(() => {
      resize()
      draw()
    })
    ro.observe(cv)

    if (reduce) {
      for (let i = 0; i < 700; i++) push()
      draw()
      const p = sample(state.current.step)
      setStats({ step: state.current.step, loss: p.loss, acc: p.acc })
      return () => ro.disconnect()
    }

    const io = new IntersectionObserver(([e]) => {
      visible = !!e?.isIntersecting
    })
    io.observe(cv)
    let uiAcc = 0

    const tick = (t: number) => {
      const dt = t - last
      last = t
      if (visible && !document.hidden) {
        acc += dt
        uiAcc += dt
        while (acc >= 50) {
          acc -= 50
          push()
        } // 20 Hz, like the real telemetry
        draw()
        if (uiAcc >= 250) {
          uiAcc = 0
          const s = state.current
          const p = sample(s.step)
          setStats({ step: s.step, loss: p.loss, acc: p.acc })
          if (s.step % 5 < 3)
            setLines((ls) => [...ls.slice(-5), { id: lineId++, step: s.step, loss: p.loss, acc: p.acc }])
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      io.disconnect()
    }
  }, [])

  return (
    <div className="ot-card ot-live" role="img" aria-label="Animated preview of a live training run">
      <div className="ot-live-head">
        <span className="ot-live-dot" /> <b>LIVE</b>
        <span className="ot-muted">Ludwig · image classifier</span>
        <span className="ot-chip">20 Hz</span>
      </div>
      <div className="ot-kpis">
        <div>
          <small>STEP</small>
          <b className="ot-tnum">{stats.step.toLocaleString()}</b>
        </div>
        <div>
          <small>LOSS</small>
          <b className="ot-tnum" style={{ color: '#00afef' }}>
            {stats.loss.toFixed(4)}
          </b>
        </div>
        <div>
          <small>ACCURACY</small>
          <b className="ot-tnum" style={{ color: '#f59e0b' }}>
            {(stats.acc * 100).toFixed(1)}%
          </b>
        </div>
      </div>
      <canvas ref={canvas} className="ot-live-canvas" />
      <div className="ot-console" style={{ fontFamily: MONO_STACK }}>
        {lines.map((l) => (
          <div key={l.id} className="ot-log">
            <span className="ot-muted">INFO</span>{' '}
            <span style={{ color: '#00afef' }}>step {String(l.step).padStart(5)}</span> loss=<b>{l.loss.toFixed(4)}</b>{' '}
            acc={l.acc.toFixed(3)}
          </div>
        ))}
        {lines.length === 0 && <div className="ot-log ot-muted">waiting for first batch…</div>}
      </div>
    </div>
  )
}

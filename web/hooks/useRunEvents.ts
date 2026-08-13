/**
 * Live training console — subscribes to GET /api/runs/:runId/events (SSE)
 * and reduces the event stream into chart-ready metric series and a capped
 * log ring buffer.
 *
 * Reconnect/replay is handled by the browser's native EventSource: it
 * remembers the last event id and sends it back as `Last-Event-ID` on
 * reconnect, and the gateway's SSE route resumes the JetStream consumer
 * from exactly that sequence — so a page refresh mid-run replays history
 * instead of losing it.
 */

import type { TrainingStatus } from '@public/store/types'
import { useEffect, useRef, useState } from 'react'

type RunEvent =
  | { kind: 'status'; runId: string; ts: string; status: TrainingStatus; message?: string }
  | {
      kind: 'metric'
      runId: string
      ts: string
      epoch: number
      split: 'train' | 'validation' | 'test'
      metrics: Record<string, number>
    }
  | { kind: 'log'; runId: string; ts: string; level: 'info' | 'warn' | 'error'; line: string }

export interface MetricPoint {
  epoch: number
  [seriesKey: string]: number
}

export interface LogLine {
  ts: string
  level: 'info' | 'warn' | 'error'
  line: string
}

export interface RunEventsState {
  status: TrainingStatus | null
  failedMessage: string | null
  /** One point per epoch, columns named `${split}.${metricName}` for @mantine/charts. */
  metricPoints: MetricPoint[]
  logs: LogLine[]
  isConnected: boolean
}

const MAX_LOG_LINES = 2000
const TERMINAL_STATUSES = new Set<TrainingStatus>(['succeeded', 'failed', 'canceled'])

/**
 * Stream a run's live status/metric/log events. Pass `active` so a
 * finished run doesn't open a connection at all.
 */
export function useRunEvents(runId: string | undefined, active: boolean, onTerminal?: () => void) {
  const [state, setState] = useState<RunEventsState>({
    status: null,
    failedMessage: null,
    metricPoints: [],
    logs: [],
    isConnected: false,
  })
  const onTerminalRef = useRef(onTerminal)
  onTerminalRef.current = onTerminal

  useEffect(() => {
    if (!runId || !active) return

    setState({ status: null, failedMessage: null, metricPoints: [], logs: [], isConnected: false })

    const source = new EventSource(`/api/runs/${runId}/events`, { withCredentials: true })

    source.onopen = () => setState((s) => ({ ...s, isConnected: true }))
    source.onerror = () => setState((s) => ({ ...s, isConnected: false }))

    const handleEvent = (raw: MessageEvent<string>) => {
      const event = JSON.parse(raw.data) as RunEvent

      setState((s) => {
        switch (event.kind) {
          case 'status': {
            if (TERMINAL_STATUSES.has(event.status)) onTerminalRef.current?.()
            return { ...s, status: event.status, failedMessage: event.message ?? s.failedMessage }
          }
          case 'metric': {
            const points = [...s.metricPoints]
            let point = points.find((p) => p.epoch === event.epoch)
            if (!point) {
              point = { epoch: event.epoch }
              points.push(point)
              points.sort((a, b) => a.epoch - b.epoch)
            }
            for (const [name, value] of Object.entries(event.metrics)) {
              point[`${event.split}.${name}`] = value
            }
            return { ...s, metricPoints: points }
          }
          case 'log': {
            const logs = [...s.logs, { ts: event.ts, level: event.level, line: event.line }]
            if (logs.length > MAX_LOG_LINES) logs.splice(0, logs.length - MAX_LOG_LINES)
            return { ...s, logs }
          }
        }
      })
    }

    source.addEventListener('status', handleEvent)
    source.addEventListener('metric', handleEvent)
    source.addEventListener('log', handleEvent)

    return () => {
      source.close()
    }
  }, [runId, active])

  return state
}

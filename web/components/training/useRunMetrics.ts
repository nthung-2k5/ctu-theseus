import type { RunEventsState } from '@public/hooks/useRunEvents'
import { useTrainingRunDetail } from '@public/lib/queries'
import type { SplitType } from '@public/store/types'
import { useMemo } from 'react'

export interface FlatMetric {
  epoch: number
  split: SplitType
  metricName: string
  value: number
}

export const SPLIT_ORDER: SplitType[] = ['train', 'validation', 'test']

/**
 * Normalises whichever metric source applies to a run into one flat, uniform shape: the live SSE
 * points while the run is active, or the persisted `TrainingMetric` rows (the full epoch history)
 * once it has finished. Chart and table code never branches on run status.
 */
export function useRunMetrics(runId: string, isActive: boolean, live: RunEventsState) {
  const { data, isLoading } = useTrainingRunDetail(runId, !isActive)
  const finishedMetrics = data?.run.metrics

  const flat: FlatMetric[] = useMemo(() => {
    if (isActive) {
      const out: FlatMetric[] = []
      for (const point of live.metricPoints) {
        for (const [key, value] of Object.entries(point)) {
          if (key === 'epoch') continue
          const dot = key.indexOf('.')
          out.push({ epoch: point.epoch, split: key.slice(0, dot) as SplitType, metricName: key.slice(dot + 1), value })
        }
      }
      return out
    }
    return (finishedMetrics ?? []).map((m) => ({
      epoch: m.epoch,
      split: m.split,
      metricName: m.metricName,
      value: m.metricValue,
    }))
  }, [isActive, live.metricPoints, finishedMetrics])

  return useMemo(() => {
    const index = new Map<string, number>()
    for (const f of flat) index.set(`${f.epoch}|${f.split}|${f.metricName}`, f.value)
    const epochs = Array.from(new Set(flat.map((f) => f.epoch))).sort((a, b) => a - b)
    const splits = SPLIT_ORDER.filter((s) => flat.some((f) => f.split === s))
    const metricNames = Array.from(new Set(flat.map((f) => f.metricName))).sort()

    /** One row per epoch with a column per split, for a single metric — the shape LineChart wants. */
    const series = (metric: string) =>
      epochs.map((epoch) => {
        const row: Record<string, number> = { epoch }
        for (const split of splits) {
          const v = index.get(`${epoch}|${split}|${metric}`)
          if (v !== undefined) row[split] = v
        }
        return row
      })

    const latest = (split: SplitType, metric: string): number | null => {
      for (let i = epochs.length - 1; i >= 0; i--) {
        const v = index.get(`${epochs[i]}|${split}|${metric}`)
        if (v !== undefined) return v
      }
      return null
    }

    const best = (split: SplitType, metric: string): number | null => {
      let min: number | null = null
      for (const e of epochs) {
        const v = index.get(`${e}|${split}|${metric}`)
        if (v !== undefined && (min === null || v < min)) min = v
      }
      return min
    }

    return { flat, index, epochs, splits, metricNames, series, latest, best, isLoading }
  }, [flat, isLoading])
}

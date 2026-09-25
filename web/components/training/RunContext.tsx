import type { RunEventsState } from '@public/hooks/useRunEvents'
import type { ProjectDetail, TrainingRunSummary } from '@public/store/types'
import { createContext, useContext } from 'react'

export interface RunContextValue {
  projectId: string
  project: ProjectDetail
  run: TrainingRunSummary
  /** queued or running: the SSE stream is (or should be) open. */
  isActive: boolean
  /** Live status from SSE when available, else the persisted one. */
  status: TrainingRunSummary['status']
  /** The one SSE subscription for the run, shared by every tab so it survives tab switches. */
  live: RunEventsState
}

const RunContext = createContext<RunContextValue | null>(null)

export const RunProvider = RunContext.Provider

export function useRunContext(): RunContextValue {
  const ctx = useContext(RunContext)
  if (!ctx) throw new Error('useRunContext must be used inside the run layout')
  return ctx
}

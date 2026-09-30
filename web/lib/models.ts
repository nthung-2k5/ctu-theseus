import { useMemo } from 'react'
import { useListProjectTrainingBackends } from './api/generated/training/training'

/** Until the backends load, assume the two names in use (Ludwig's, and the generic default). */
const FALLBACK_PARAM_NAMES = ['encoderId', 'modelId']

/**
 * The hyperparameter keys that carry a run's MODEL choice, as each trainer backend names it
 * (`modelParamName`: Ludwig keeps "encoderId", others use "modelId"). Everything else in a run's
 * hyperparameters is a tunable knob, so views that list the knobs leave these keys out and show the
 * model by name instead.
 */
export function useModelParamNames(projectId: string): ReadonlySet<string> {
  const { data } = useListProjectTrainingBackends(projectId)
  return useMemo(() => new Set(data ? data.backends.map((b) => b.modelParamName) : FALLBACK_PARAM_NAMES), [data])
}

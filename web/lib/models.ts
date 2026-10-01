import { useMemo } from 'react'
import { useListProjectTrainingBackends } from './api/generated/training/training'

/** Until the backends load, assume the two names in use (Ludwig's, and the generic default). */
const FALLBACK_PARAM_NAMES = ['encoderId', 'modelId']

/**
 * Display label for each hyperparameter name, taken from the trainer backends' own parameter specs, so a
 * run's configuration reads the same as the form it was started from ("Freeze Backbone", not `freezeBackbone`).
 * Empty until the backends load, and has no entry for a key no current spec declares (an old run's).
 */
export function useHyperparamLabels(projectId: string): ReadonlyMap<string, string> {
  const { data } = useListProjectTrainingBackends(projectId)
  return useMemo(() => {
    const labels = new Map<string, string>()
    for (const backend of data?.backends ?? []) {
      for (const param of backend.params) if (!labels.has(param.name)) labels.set(param.name, param.label)
    }
    return labels
  }, [data])
}

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

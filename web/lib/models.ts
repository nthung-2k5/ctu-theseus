import { useMemo } from 'react'
import { useListProjectTrainingBackends } from './api/generated/training/training'
import { hyperparamLabel } from './format'

/** Until the backends load, assume the two names in use (Ludwig's, and the generic default). */
const FALLBACK_PARAM_NAMES = ['encoderId', 'modelId']

/**
 * Turns a hyperparameter key into its display name wherever a run's configuration is listed (the Config tab, a
 * sweep's trials, the run comparison), so all of them read like the form the run was started from
 * ("Freeze Backbone", not `freezeBackbone`). Labels come from the trainer backends' own parameter specs, with
 * `hyperparamLabel`'s fallbacks for what they do not name (an old run's keys, or before the backends load).
 */
export function useHyperparamLabel(projectId: string): (key: string) => string {
  const { data } = useListProjectTrainingBackends(projectId)
  return useMemo(() => {
    const specLabels = new Map<string, string>()
    for (const backend of data?.backends ?? []) {
      for (const param of backend.params) if (!specLabels.has(param.name)) specLabels.set(param.name, param.label)
    }
    return (key: string) => hyperparamLabel(key, specLabels)
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

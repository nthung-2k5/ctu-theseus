import type { TrainingRunSummary } from '@public/store/types'

/**
 * A run's headline score: accuracy for a classifier, mAP50 for a detector, R² for a regressor. The server picks it
 * from the evaluation report (`services/evaluation_score.py`), so nothing here hard-codes a metric.
 */
export type RunScore = NonNullable<NonNullable<TrainingRunSummary['evaluation']>['score']>

export const scoreOf = (run: TrainingRunSummary): RunScore | null =>
  run.evaluation?.status === 'success' ? (run.evaluation.score ?? null) : null

/** Every run of a project trains the same task, so they share one metric; the first run that has a score names it. */
export const headlineScore = (runs: TrainingRunSummary[]): RunScore | null =>
  runs.map(scoreOf).find((score) => score != null) ?? null

/** Higher is always better here, so a lower-is-better metric (an error) is negated; no score is worst of all. */
export function goodness(run: TrainingRunSummary, higherIsBetter: boolean): number {
  const score = scoreOf(run)
  if (score == null) return Number.NEGATIVE_INFINITY
  return higherIsBetter ? score.value : -score.value
}

/** The run with the best score, or null while none has been evaluated. */
export function bestRun(runs: TrainingRunSummary[]): TrainingRunSummary | null {
  const higherIsBetter = headlineScore(runs)?.higherIsBetter ?? true
  let best: TrainingRunSummary | null = null
  let bestGoodness = Number.NEGATIVE_INFINITY
  for (const run of runs) {
    const g = goodness(run, higherIsBetter)
    if (g > bestGoodness) {
      best = run
      bestGoodness = g
    }
  }
  return best
}

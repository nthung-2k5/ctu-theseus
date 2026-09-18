/**
 * NATS subject/stream topology shared with the Python worker — single
 * source of truth. Mirrored into `ai_service/schema/subjects.py` by
 * `server/compile_schema.ts`; edit here, then re-run that script so the two
 * sides can't drift out of hand-sync the way they used to.
 *
 * `{token}` placeholders are filled with `fillSubject` below on the TS side,
 * and with Python's `str.format(**kwargs)` on the generated Python side —
 * both use the exact same `{token}` syntax, so the templates themselves
 * don't need translating.
 */

export const SUBJECT_TEMPLATES = {
  trainTask: 'theseus.task.train.{runId}',
  exportTask: 'theseus.task.export.{jobId}',
  abortCommand: 'theseus.command.run.{runId}',
  runEvent: 'theseus.event.run.{runId}.{kind}',
  runEventsWildcard: 'theseus.event.run.{runId}.>',
  // Dispatched onto THESEUS_TASKS (already covers theseus.task.> — no
  // subjects-array change needed there), so inference inherits max_deliver,
  // ack windows and the DLQ from the same pull-consumer machinery train and
  // export use, instead of the unbounded 10s-timeout core-NATS request/reply
  // the sync version used.
  inferenceTask: 'theseus.task.inference.{inferenceId}',
  // Fire-and-forget preload signal — distinct top-level second token
  // ('inference' vs 'task') from inferenceTask, so the two can never collide.
  inferenceWarm: 'theseus.inference.warm.{runId}',
  // Terminal result of a dispatched inference task — last-value-per-subject
  // (see THESEUS_INFERENCE_RESULTS below), polled by GET
  // /api/inference/:runId/jobs/:inferenceId. No message yet means "pending":
  // there is deliberately no explicit queued/running value written.
  inferenceResult: 'theseus.inference.result.{inferenceId}',
  dlq: 'theseus.dlq.{kind}.{id}',
  abortFlag: 'theseus.abortflag.{runId}',
} as const

export const SUBJECT_WILDCARDS = {
  tasks: 'theseus.task.>',
  events: 'theseus.event.>',
  allRunEvents: 'theseus.event.run.*.>',
  commands: 'theseus.command.>',
  dlq: 'theseus.dlq.>',
  abortFlags: 'theseus.abortflag.>',
  trainTasks: 'theseus.task.train.*',
  exportTasks: 'theseus.task.export.*',
  inferenceTasks: 'theseus.task.inference.*',
  commandsPerRun: 'theseus.command.run.*',
  inferenceWarmAll: 'theseus.inference.warm.*',
  inferenceResultsAll: 'theseus.inference.result.*',
} as const

export type RetentionKind = 'workqueue' | 'limits'

export interface StreamDef {
  name: string
  subjects: readonly string[]
  retention: RetentionKind
  maxAgeSeconds: number
  /** Only set for the last-value-per-subject abort-flag stream. */
  maxMsgsPerSubject?: number
}

export const STREAM_DEFS: readonly StreamDef[] = [
  { name: 'THESEUS_TASKS', subjects: [SUBJECT_WILDCARDS.tasks], retention: 'workqueue', maxAgeSeconds: 24 * 3600 },
  { name: 'THESEUS_EVENTS', subjects: [SUBJECT_WILDCARDS.events], retention: 'limits', maxAgeSeconds: 7 * 24 * 3600 },
  { name: 'THESEUS_COMMANDS', subjects: [SUBJECT_WILDCARDS.commands], retention: 'workqueue', maxAgeSeconds: 3600 },
  // Permanently-failed tasks/commands land here (see lib/nats.ts `subscribe`)
  // instead of retrying forever or silently expiring off THESEUS_TASKS.
  { name: 'THESEUS_DLQ', subjects: [SUBJECT_WILDCARDS.dlq], retention: 'limits', maxAgeSeconds: 30 * 24 * 3600 },
  // One "last value wins" message per run — the same last-value-per-subject
  // mechanism NATS's own KV feature is built on, used directly here so
  // abort intent survives a worker restart without pulling in a separate
  // KV client package. See lib/nats.ts `setAbortFlag`/`isAborted`.
  {
    name: 'THESEUS_ABORT_FLAGS',
    subjects: [SUBJECT_WILDCARDS.abortFlags],
    retention: 'limits',
    maxAgeSeconds: 7 * 24 * 3600,
    maxMsgsPerSubject: 1,
  },
  // Same last-value-per-subject idiom as THESEUS_ABORT_FLAGS. Self-expires
  // after an hour so nothing needs to sweep it — the durable result lives
  // in Postgres (`inferenceJobs`, see lib/microservice.ts's
  // startInferenceResultsConsumer), which persists it the moment it's
  // published rather than relying on this stream still holding it later.
  {
    name: 'THESEUS_INFERENCE_RESULTS',
    subjects: [SUBJECT_WILDCARDS.inferenceResultsAll],
    retention: 'limits',
    maxAgeSeconds: 3600,
    maxMsgsPerSubject: 1,
  },
] as const

function fillSubject(template: string, params: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_match, key: string) => {
    const value = params[key]
    if (value === undefined) throw new Error(`Missing subject param '${key}' for template '${template}'`)
    return value
  })
}

export const subject = {
  trainTask: (runId: string) => fillSubject(SUBJECT_TEMPLATES.trainTask, { runId }),
  exportTask: (jobId: string) => fillSubject(SUBJECT_TEMPLATES.exportTask, { jobId }),
  abortCommand: (runId: string) => fillSubject(SUBJECT_TEMPLATES.abortCommand, { runId }),
  runEvent: (runId: string, kind: string) => fillSubject(SUBJECT_TEMPLATES.runEvent, { runId, kind }),
  runEventsWildcard: (runId: string) => fillSubject(SUBJECT_TEMPLATES.runEventsWildcard, { runId }),
  inferenceTask: (inferenceId: string) => fillSubject(SUBJECT_TEMPLATES.inferenceTask, { inferenceId }),
  inferenceWarm: (runId: string) => fillSubject(SUBJECT_TEMPLATES.inferenceWarm, { runId }),
  inferenceResult: (inferenceId: string) => fillSubject(SUBJECT_TEMPLATES.inferenceResult, { inferenceId }),
  dlq: (kind: string, id: string) => fillSubject(SUBJECT_TEMPLATES.dlq, { kind, id }),
  abortFlag: (runId: string) => fillSubject(SUBJECT_TEMPLATES.abortFlag, { runId }),
}

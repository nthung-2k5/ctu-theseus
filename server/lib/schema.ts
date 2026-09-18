import { SplitTypes, TrainingStatuses } from '@server/lib/enums'
import { t } from 'elysia'

/** Gateway → worker: start a training run. */
export const TrainTaskSchema = t.Object(
  {
    runId: t.String({ format: 'uuid' }),
    projectId: t.String({ format: 'uuid' }),
    datasetVersionId: t.String({ format: 'uuid' }),
    /** S3 key of the compiled Ludwig config.yaml, in the training bucket. */
    configKey: t.String(),
    /** S3 key of the version's dataset.parquet, in the datasets bucket. */
    datasetKey: t.String(),
    /** S3 prefix the worker writes Ludwig's output_directory under. */
    outputPrefix: t.String(),
  },
  { title: 'TrainTask' },
)

/** Gateway → worker: export a trained model to a deployable format. */
export const ExportTaskSchema = t.Object(
  {
    jobId: t.String({ format: 'uuid' }),
    runId: t.String({ format: 'uuid' }),
    format: t.UnionEnum(['onnx', 'torchscript']),
    /**
     * S3 key of the run's dataset.parquet, in the datasets bucket — lets the
     * worker pull one real test-split row to verify the export against
     * (writes `expected.json` next to the artifact). Omitted if the run's
     * source snapshot is no longer available.
     */
    datasetKey: t.Optional(t.String()),
  },
  { title: 'ExportTask' },
)

/**
 * Gateway → worker: dispatch one inference job onto THESEUS_TASKS —
 * `inferenceId` is the job's own identity (distinct from `runId`, which
 * model to use), and is what the terminal result is published against (see
 * `InferenceResponseSchema` and `theseus.inference.result.{inferenceId}`
 * below). `payload` mirrors the task registry's `itemSpec.payload` (see
 * server/lib/tasks/types.ts): file-backed modalities (vision/audio) upload
 * through the NATS object store first and reference it by key; text
 * payloads carry one value per input field inline (`fields`, keyed by the
 * Ludwig input feature's column name — see `getInferenceInputSpec` — so a
 * multi-input task like question_answering can send both `context` and
 * `question`); record payloads carry the tabular row inline. `batch` scores
 * many rows in one call (CSV upload — text/tabular tasks only, one column
 * per Ludwig input feature) instead of the single-row `text`/`record`
 * payloads above, which is what makes it worth a distinct kind: a batch
 * response is a downloadable results file, not an inline `InferenceOutput`.
 */
export const InferenceTaskSchema = t.Object(
  {
    inferenceId: t.String({ format: 'uuid' }),
    runId: t.String({ format: 'uuid' }),
    /** Caps how many classes a `classification` output returns (sorted desc). Ignored for `batch`. */
    topK: t.Optional(t.Integer({ minimum: 1, maximum: 1000, default: 100 })),
    payload: t.Union([
      t.Object({ kind: t.Literal('file'), uploadKey: t.String(), uploadFilename: t.String() }),
      t.Object({ kind: t.Literal('text'), fields: t.Record(t.String(), t.String()) }),
      t.Object({ kind: t.Literal('record'), record: t.Record(t.String(), t.Union([t.String(), t.Number()])) }),
      t.Object({ kind: t.Literal('batch'), uploadKey: t.String(), uploadFilename: t.String() }),
    ]),
  },
  { title: 'InferenceTask' },
)

/**
 * Worker → gateway: the shape of a successful prediction depends on the
 * trained model's Ludwig output feature type — a flat `Record<string,
 * number>` can't represent generated text or a token-tagged sequence, so
 * this is a tagged union instead. `feature` is the output feature's name
 * (read off the loaded model, not the task registry, since that's what
 * Ludwig actually named its output column).
 */
export const InferenceOutputSchema = t.Union([
  t.Object({
    kind: t.Literal('classification'),
    feature: t.String(),
    classes: t.Array(t.Object({ label: t.String(), confidence: t.Number() })),
  }),
  t.Object({
    kind: t.Literal('regression'),
    feature: t.String(),
    value: t.Number(),
  }),
  t.Object({
    kind: t.Literal('text'),
    feature: t.String(),
    text: t.String(),
  }),
  t.Object({
    kind: t.Literal('tokens'),
    feature: t.String(),
    tokens: t.Array(t.Object({ token: t.String(), tag: t.String() })),
  }),
])

/**
 * Worker → gateway: the terminal outcome of one inference job, published to
 * `theseus.inference.result.{inferenceId}` once (see `THESEUS_INFERENCE_RESULTS`
 * in lib/subjects.ts) — either on success, or once retries are exhausted
 * (`on_inference_permanent_failure` in ai_service/tasks/inference.py). No
 * "pending"/"running" member: the poll route treats the absence of a
 * message on that subject as pending, so there's nothing to write until
 * there's a real outcome.
 *
 * `runId` is carried on every member so the poll route
 * (`routes/inference.ts`) can verify a caller's `inferenceId` actually
 * belongs to the run they're authorized for — `inferenceId` alone isn't
 * a bearer capability, since THESEUS_INFERENCE_RESULTS is keyed on it
 * without reference to who dispatched the job.
 */
export const InferenceResponseSchema = t.Union(
  [
    t.Object({ status: t.Literal('success'), runId: t.String({ format: 'uuid' }), output: InferenceOutputSchema }),
    // Distinct from 'success' (not a third variant of it) — a batch job has
    // no single InferenceOutput to inline; the result is a file, referenced
    // by key. resultKey is the BUCKET_MODELS key of the results CSV (see
    // ai_service/tasks/inference.py's batch branch and
    // server/lib/storage.ts's batchInferenceResultKey).
    t.Object({
      status: t.Literal('batch'),
      runId: t.String({ format: 'uuid' }),
      resultKey: t.String(),
      rowCount: t.Integer({ minimum: 0 }),
    }),
    t.Object({ status: t.Literal('failed'), runId: t.String({ format: 'uuid' }), error: t.String() }),
  ],
  { title: 'InferenceResponse' },
)

/**
 * Worker → gateway: everything a training run emits, in one tagged union.
 * Published to `theseus.event.run.{runId}.{kind}` on the THESEUS_EVENTS
 * stream — the JetStream sequence number becomes the SSE `id` for replay.
 */
export const RunEventSchema = t.Union(
  [
    t.Object({
      kind: t.Literal('status'),
      runId: t.String({ format: 'uuid' }),
      ts: t.String({ format: 'date-time' }),
      status: t.UnionEnum(TrainingStatuses),
      message: t.Optional(t.String()),
    }),
    t.Object({
      kind: t.Literal('metric'),
      runId: t.String({ format: 'uuid' }),
      ts: t.String({ format: 'date-time' }),
      epoch: t.Integer({ minimum: 0 }),
      split: t.UnionEnum(SplitTypes),
      metrics: t.Record(t.String(), t.Number()),
    }),
    t.Object({
      kind: t.Literal('log'),
      runId: t.String({ format: 'uuid' }),
      ts: t.String({ format: 'date-time' }),
      level: t.UnionEnum(['info', 'warn', 'error']),
      line: t.String(),
    }),
    t.Object({
      kind: t.Literal('export'),
      runId: t.String({ format: 'uuid' }),
      ts: t.String({ format: 'date-time' }),
      jobId: t.String({ format: 'uuid' }),
      status: t.UnionEnum(['success', 'failed']),
      format: t.Optional(t.UnionEnum(['onnx', 'torchscript'])),
      exportKey: t.Optional(t.String()),
      error: t.Optional(t.String()),
    }),
    t.Object({
      kind: t.Literal('evaluation'),
      runId: t.String({ format: 'uuid' }),
      ts: t.String({ format: 'date-time' }),
      status: t.UnionEnum(['success', 'failed']),
      split: t.Optional(t.UnionEnum(['train', 'validation', 'test', 'full'])),
      /** S3 key (BUCKET_TRAINING) of the bounded evaluation report.json — see server/lib/microservice.ts's 'evaluation' case. */
      reportKey: t.Optional(t.String()),
      /** S3 key (BUCKET_TRAINING) of the full per-row predictions parquet — not ingested by the gateway, download-only. */
      predictionsKey: t.Optional(t.String()),
      /** Headline metric for quick display without fetching the full report — accuracy (classification) or R² (regression). */
      headlineMetric: t.Optional(t.Number()),
      error: t.Optional(t.String()),
    }),
  ],
  { title: 'RunEvent' },
)

/** Gateway → worker: control a running or queued job. */
export const CommandSchema = t.Object(
  {
    command: t.Literal('abort'),
    runId: t.String({ format: 'uuid' }),
  },
  { title: 'Command' },
)

/** Schemas compiled to `schema/{name}.json` and mirrored into ai_service/schema/{name}.py. */
export const compiledSchemas = [
  [TrainTaskSchema, 'train_task'],
  [ExportTaskSchema, 'export_task'],
  [InferenceTaskSchema, 'inference_task'],
  [InferenceResponseSchema, 'inference_response'],
  [RunEventSchema, 'run_event'],
  [CommandSchema, 'command'],
] as const

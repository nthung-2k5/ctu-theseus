/**
 * NATS message contracts shared with the Python worker.
 */

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
  },
  { title: 'ExportTask' },
)

/** Gateway ↔ worker: synchronous inference request/response over core NATS. */
export const InferenceRequestSchema = t.Object(
  {
    runId: t.String({ format: 'uuid' }),
    uploadKey: t.String(),
    uploadFilename: t.String(),
    threshold: t.Number({ minimum: 0, maximum: 1, default: 0.5 }),
  },
  { title: 'InferenceRequest' },
)

export const InferenceResponseSchema = t.Union(
  [
    t.Object({ status: t.Literal('success'), results: t.Record(t.String(), t.Number()) }),
    t.Object({ status: t.Literal('failed'), error: t.String() }),
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
  [InferenceRequestSchema, 'inference_request'],
  [RunEventSchema, 'run_event'],
  [CommandSchema, 'command'],
] as const

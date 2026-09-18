import {
  AnnotationTypes,
  AudioCodecs,
  DatasetModalities,
  DatasetVersionStatuses,
  EvaluationSplits,
  EvaluationStatuses,
  ExportFormats,
  ExportLangs,
  ExportStatuses,
  ExportTiers,
  ImageFormats,
  InferenceJobStatuses,
  ProjectTasks,
  SplitTypes,
  SweepStatuses,
  SweepStrategies,
  TrainingStatuses,
} from '@server/lib/enums'
import { sql } from 'drizzle-orm'
import {
  boolean,
  char,
  check,
  index,
  integer,
  jsonb,
  numeric,
  pgEnum,
  pgTable,
  primaryKey,
  real,
  text,
  timestamp,
  unique,
  uniqueIndex,
  uuid,
  varchar,
} from 'drizzle-orm/pg-core'

/* ------------------------------------------------------------------ */
/*  Users                                                             */
/* ------------------------------------------------------------------ */
export const users = pgTable('users', {
  id: uuid('id').primaryKey().default(sql`uuidv7()`), // Wait until DrizzleORM merges PR#5722 (https://github.com/drizzle-team/drizzle-orm/pull/5722)
  name: text('name').notNull(),
  email: text('email').notNull().unique(),
  emailVerified: boolean('email_verified').default(false).notNull(),
  image: text('image'),
  createdAt: timestamp('created_at').notNull(),
  updatedAt: timestamp('updated_at')
    .$onUpdate(() => new Date())
    .notNull(),
  role: text('role'),
  banned: boolean('banned').default(false),
  banReason: text('ban_reason'),
  banExpires: timestamp('ban_expires'),
})

/* ------------------------------------------------------------------ */
/*  Sessions                                                          */
/* ------------------------------------------------------------------ */
export const sessions = pgTable(
  'sessions',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    expiresAt: timestamp('expires_at').notNull(),
    token: text('token').notNull().unique(),
    createdAt: timestamp('created_at').notNull(),
    updatedAt: timestamp('updated_at')
      .$onUpdate(() => new Date())
      .notNull(),
    ipAddress: text('ip_address'),
    userAgent: text('user_agent'),
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    impersonatedBy: text('impersonated_by'),
  },
  (table) => [index('sessions_userId_idx').on(table.userId)],
)

export const accounts = pgTable(
  'accounts',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    accountId: text('account_id').notNull(),
    providerId: text('provider_id').notNull(),
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    accessToken: text('access_token'),
    refreshToken: text('refresh_token'),
    idToken: text('id_token'),
    accessTokenExpiresAt: timestamp('access_token_expires_at'),
    refreshTokenExpiresAt: timestamp('refresh_token_expires_at'),
    scope: text('scope'),
    password: text('password'),
    createdAt: timestamp('created_at').notNull(),
    updatedAt: timestamp('updated_at')
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [index('accounts_userId_idx').on(table.userId)],
)

export const verifications = pgTable(
  'verifications',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    identifier: text('identifier').notNull(),
    value: text('value').notNull(),
    expiresAt: timestamp('expires_at').notNull(),
    createdAt: timestamp('created_at').notNull(),
    updatedAt: timestamp('updated_at')
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [index('verifications_identifier_idx').on(table.identifier)],
)

/* ------------------------------------------------------------------ */
/*  API Keys                                                          */
/*                                                                     */
/*  Bearer credentials for the hosted prediction API (POST             */
/*  /api/v1/predict/:runId — see routes/api-v1.ts and the `apiKey`     */
/*  macro in routes/auth.ts). Only a sha256 hash of the key is ever    */
/*  stored — the plaintext is shown to the user exactly once, at       */
/*  creation, and is unrecoverable after that. `keyPrefix` is stored   */
/*  purely so a user can tell which key is which in a list without     */
/*  ever seeing the secret again.                                      */
/* ------------------------------------------------------------------ */
export const apiKeys = pgTable(
  'api_keys',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    name: varchar('name', { length: 100 }).notNull(),
    // sha256 hex digest of the raw key — never the raw key itself.
    keyHash: char('key_hash', { length: 64 }).notNull().unique(),
    // First few characters of the raw key, e.g. "thsk_ab12" — display only,
    // not a security boundary (the hash is what's actually checked).
    keyPrefix: varchar('key_prefix', { length: 16 }).notNull(),
    lastUsedAt: timestamp('last_used_at', { withTimezone: true }),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    // Soft-revoked rather than deleted — keeps the audit trail of what a
    // (possibly leaked) key was and when it stopped being valid.
    revokedAt: timestamp('revoked_at', { withTimezone: true }),
  },
  (table) => [index('apiKeys_userId_idx').on(table.userId)],
)

/* ------------------------------------------------------------------ */
/*  Projects                                                          */
/* ------------------------------------------------------------------ */
export const projectTaskEnum = pgEnum('project_task', ProjectTasks)

export const projects = pgTable(
  'projects',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    name: text('name').notNull(),
    description: text('description').default(''),
    task: projectTaskEnum('task').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [index('projects_userId_idx').on(table.userId)],
)

/* ------------------------------------------------------------------ */
/*  Dataset                                                           */
/* ------------------------------------------------------------------ */
export const modalityEnum = pgEnum('modality', DatasetModalities)
export const splitTypeEnum = pgEnum('split_type', SplitTypes)
export const imageFormatEnum = pgEnum('image_format', ImageFormats)
export const audioCodecEnum = pgEnum('audio_codec', AudioCodecs)
export const annotationTypeEnum = pgEnum('annotation_type', AnnotationTypes)
export const datasetVersionStatusEnum = pgEnum('dataset_version_status', DatasetVersionStatuses)

// --- 1. DATASET REGISTRY ---

export const datasets = pgTable('datasets', {
  projectId: uuid('project_id')
    .notNull()
    .primaryKey()
    .references(() => projects.id, { onDelete: 'cascade' }),
  modality: modalityEnum('modality').notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true })
    .$onUpdate(() => new Date())
    .notNull(),
})

export const datasetVersions = pgTable(
  'dataset_versions',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    datasetId: uuid('dataset_id')
      .notNull()
      .references(() => datasets.projectId, { onDelete: 'cascade' }),

    // If NULL => draft dataset (project's working copy)
    // If NOT NULL => immutable snapshot (used for training runs)
    versionTag: varchar('version_tag', { length: 50 }),

    // The draft row is permanently 'draft'. Snapshots move
    // building -> ready|failed as the parquet is built.
    status: datasetVersionStatusEnum('status').default('draft').notNull(),
    itemCount: integer('item_count'),
    classCount: integer('class_count'),
    parquetKey: text('parquet_key'),
    failedMessage: text('failed_message'),
    builtAt: timestamp('built_at', { withTimezone: true }),

    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index('datasetVersions_datasetId_idx').on(table.datasetId),
    unique('datasetVersions_datasetId_versionTag_key').on(table.datasetId, table.versionTag),
  ],
)

// --- 2. CENTRAL DATA ITEM TABLE (the project-wide deduplicated pool) ---

export const datasetItems = pgTable(
  'dataset_items',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    datasetId: uuid('dataset_id')
      .notNull()
      .references(() => datasets.projectId, { onDelete: 'cascade' }),
    externalId: varchar('external_id', { length: 255 }), // ID mapping back to source cloud storage or local path
    storageUrl: text('storage_url'), // Path to raw files (S3, GCS) if unstructured (images/audio/raw text files)
    contentHash: char('content_hash', { length: 64 }), // sha256 of the raw bytes, for pool dedup
    byteSize: integer('byte_size'),
    // embedding: vector('embedding', { dimensions: 1536 }), // Vector representation of the data item for semantic filtering/filtering out duplicates
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    // Set instead of hard-deleting when a snapshot's RESTRICT FK blocks the
    // delete (see dataset_version_items below) — the row (and its feature
    // rows) must stay intact for that snapshot, but it's gone from the pool.
    deletedAt: timestamp('deleted_at', { withTimezone: true }),
  },
  (table) => [
    index('dataset_items_datasetId_idx').on(table.datasetId),
    unique('dataset_items_datasetId_contentHash_key').on(table.datasetId, table.contentHash),
    // TODO: Add embedding index when we decide to use it
    // index('dataset_items_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops')),
  ],
)

// --- 2b. VERSION <-> ITEM MEMBERSHIP (a snapshot is a set of pool items + their split) ---

export const datasetVersionItems = pgTable(
  'dataset_version_items',
  {
    versionId: uuid('version_id')
      .notNull()
      .references(() => datasetVersions.id, { onDelete: 'cascade' }),
    // restrict: an item referenced by any snapshot can't be deleted from the
    // pool — this is what makes a snapshot genuinely immutable.
    itemId: uuid('item_id')
      .notNull()
      .references(() => datasetItems.id, { onDelete: 'restrict' }),
    splitType: splitTypeEnum('split_type').notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.versionId, table.itemId] }),
    index('datasetVersionItems_versionId_splitType_idx').on(table.versionId, table.splitType),
  ],
)

// --- 3. MODALITY-SPECIFIC SUB-TABLES (Extends dataset_items via 1-to-1 relationships) ---

// A. Text Modality (LLM Pretraining, SFT, Translation, NER)
export const textFeatures = pgTable(
  'text_features',
  {
    itemId: uuid('item_id')
      .primaryKey()
      .references(() => datasetItems.id, { onDelete: 'cascade' }),
    rawText: text('raw_text').notNull(),
    tokenCount: integer('token_count'),
    languageCode: varchar('language_code', { length: 10 }),
    metaJson: jsonb('meta_json'),
  },
  (table) => [index('idx_text_features_lang').on(table.languageCode)],
)

// B. Vision Modality (Classification, Object Detection, Segmentation)
export const visionFeatures = pgTable('vision_features', {
  itemId: uuid('item_id')
    .primaryKey()
    .references(() => datasetItems.id, { onDelete: 'cascade' }),
  width: integer('width').notNull(),
  height: integer('height').notNull(),
  channels: integer('channels').default(3),
  imageFormat: imageFormatEnum('image_format'),
  exifData: jsonb('exif_data'),
})

// C. Audio Modality (ASR, Text-to-Speech, Audio Classification)
export const audioFeatures = pgTable('audio_features', {
  itemId: uuid('item_id')
    .primaryKey()
    .references(() => datasetItems.id, { onDelete: 'cascade' }),
  durationSeconds: numeric('duration_seconds', { precision: 8, scale: 3 }).notNull(),
  sampleRateHz: integer('sample_rate_hz').notNull(), // 16000, 44100, 48000
  channels: integer('channels').default(1),
  audioCodec: audioCodecEnum('audio_codec'),
})

// D. Tabular & Structured Modality (XGBoost, Deep Learning Tabular)
export const tabularFeatures = pgTable(
  'tabular_features',
  {
    itemId: uuid('item_id')
      .primaryKey()
      .references(() => datasetItems.id, { onDelete: 'cascade' }),
    featuresJson: jsonb('features_json').notNull(), // Stores key-value features, handling variable schemas dynamically
  },
  (table) => [
    // GIN index for highly performant JSONB key-value querying
    index('idx_tabular_features_gin').using('gin', table.featuresJson),
  ],
)

/* ------------------------------------------------------------------ */
/*  GLOBAL ANNOTATIONS & GROUND TRUTH LABELS                          */
/* ------------------------------------------------------------------ */
export const labelClasses = pgTable(
  'label_classes',
  {
    classId: uuid('class_id').primaryKey().default(sql`uuidv7()`),
    datasetId: uuid('dataset_id')
      .references(() => datasets.projectId, { onDelete: 'cascade' })
      .notNull(),

    // The human-readable name (e.g., "pedestrian", "positive_sentiment")
    name: varchar('name', { length: 100 }).notNull(),

    // Optional description so human annotators know exactly what this class means
    description: text('description'),

    // Helpful for UI rendering (e.g., drawing bounding boxes in a specific color)
    uiColorHex: varchar('ui_color_hex', { length: 7 }).default('#FFFFFF'),

    // Soft delete flag
    isActive: boolean('is_active').default(true).notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  },
  (table) => [
    index('idx_label_classes_dataset').on(table.datasetId),
    unique('labelClasses_datasetId_name_key').on(table.datasetId, table.name),
  ],
)

export const annotations = pgTable(
  'annotations',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    itemId: uuid('item_id')
      .notNull()
      .references(() => datasetItems.id, { onDelete: 'cascade' }),
    annotatorId: varchar('annotator_id', { length: 100 }),
    annotationType: annotationTypeEnum('annotation_type').notNull(),

    // Polymorphic label containers
    classId: uuid('class_id').references(() => labelClasses.classId, { onDelete: 'restrict' }),
    labelTextSequence: text('label_text_sequence'), // For Seq2Seq, target translation, or text responses
    labelStructured: jsonb('label_structured'), // For Vision tasks (JSON structures containing arrays of objects), e.g., Bounding Box: [{"class": "cat", "bbox": [x, y, w, h]}]

    confidenceScore: numeric('confidence_score', { precision: 4, scale: 3 }), // Confidence score of the annotation (0.0 - 1.0)
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(), // Timestamp of when the annotation was created
  },
  (table) => [
    index('idx_annotations_item').on(table.itemId),
    index('idx_annotations_type').on(table.annotationType),
    // The item-list class filter (routes/datasets.ts) selects annotations by
    // classId + annotationType on every filtered page load; without this it
    // scanned the whole table.
    index('idx_annotations_class').on(table.classId, table.annotationType),
    // At most one classification label per item. Without it, two concurrent
    // classify calls both read `existing == null` and both insert, and the
    // snapshot builder then picks one arbitrarily — baking a nondeterministic
    // ground-truth label into the parquet.
    uniqueIndex('annotations_item_classification_key')
      .on(table.itemId)
      .where(sql`${table.annotationType} = 'classification'`),
    // Drizzle table-level check constraint for confidence limits
    check('confidence_bounds', sql`${table.confidenceScore} BETWEEN 0.0 AND 1.0`),
  ],
)

/* ------------------------------------------------------------------ */
/*  Sweeps                                                            */
/*                                                                     */
/*  A hyperparameter sweep is orchestrated entirely on top of the      */
/*  existing training pipeline — no separate execution engine. Each   */
/*  trial is an ordinary trainingRuns row (sweepId + trialIndex below) */
/*  dispatched through the same queueTraining() every manually-started */
/*  run goes through, so it inherits the run/metric/cancel/DLQ         */
/*  machinery for free. See server/lib/sweep.ts for search-space       */
/*  expansion and server/routes/sweeps.ts for dispatch.                */
/* ------------------------------------------------------------------ */
export const sweepStrategyEnum = pgEnum('sweep_strategy', SweepStrategies)
export const sweepStatusEnum = pgEnum('sweep_status', SweepStatuses)

export const sweeps = pgTable(
  'sweeps',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    projectId: uuid('project_id')
      .notNull()
      .references(() => projects.id, { onDelete: 'cascade' }),
    datasetVersionId: uuid('dataset_version_id')
      .notNull()
      .references(() => datasetVersions.id, { onDelete: 'cascade' }),
    name: varchar('name', { length: 255 }).notNull(),
    // { [trainerKnob]: candidateValue[] } — see server/lib/sweep.ts's SweepSearchSpace.
    searchSpace: jsonb('search_space').notNull(),
    strategy: sweepStrategyEnum('strategy').notNull(),
    maxTrials: integer('max_trials').notNull(),
    // 'running' is the only status ever written at creation; 'completed' is
    // set lazily (GET /sweeps/:sweepId reconciles it once every trial run
    // has reached a terminal status — same pattern as the export route's
    // reconcileConverting), and 'canceled' is written explicitly by the
    // cancel route. See enums.ts's SweepStatuses for why these are distinct.
    status: sweepStatusEnum('status').default('running').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [index('sweeps_projectId_idx').on(table.projectId)],
)

/* ------------------------------------------------------------------ */
/*  Training Runs                                                     */
/* ------------------------------------------------------------------ */
export const trainingStatusEnum = pgEnum('training_status', TrainingStatuses)

export const trainingRuns = pgTable(
  'training_runs',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    name: varchar('name', { length: 255 }).notNull(),
    status: trainingStatusEnum().default('queued').notNull(),
    projectId: uuid('project_id')
      .notNull()
      .references(() => projects.id, { onDelete: 'cascade' }),

    // Provenance
    datasetVersionId: uuid('dataset_version_id')
      .notNull()
      .references(() => datasetVersions.id, { onDelete: 'cascade' }),

    // Set only for a run dispatched as one trial of a sweep (see `sweeps`
    // above) — NULL for an ordinary manually-started run.
    sweepId: uuid('sweep_id').references(() => sweeps.id, { onDelete: 'cascade' }),
    trialIndex: integer('trial_index'),

    // User-facing hyperparameter selections (trainer knobs, encoder choice, ...)
    hyperparameters: jsonb('hyperparameters').notNull(),
    // The exact compiled Ludwig config sent to the worker, for reproducibility
    ludwigConfig: jsonb('ludwig_config'),
    // S3 key of the compiled config.yaml
    configKey: text('config_key'),
    bestEpoch: integer('best_epoch'),
    // Last time the worker reported progress; lets the gateway detect and
    // fail orphaned runs after a restart.
    heartbeatAt: timestamp('heartbeat_at', { withTimezone: true }),

    startedAt: timestamp('started_at', { withTimezone: true }),
    completedAt: timestamp('completed_at', { withTimezone: true }),
    failedMessage: text('failed_message'),

    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [
    index('trainingRuns_projectId_idx').on(table.projectId),
    index('trainingRuns_sweepId_idx').on(table.sweepId),
  ],
)

/* ------------------------------------------------------------------ */
/*  Training Metrics                                                  */
/* ------------------------------------------------------------------ */
export const trainingMetrics = pgTable(
  'training_metrics',
  {
    trainingRunId: uuid('training_run_id')
      .notNull()
      .references(() => trainingRuns.id, { onDelete: 'cascade' }),
    epoch: integer('epoch').notNull(),
    split: splitTypeEnum('split').notNull(),
    metricName: varchar('metric_name', { length: 64 }).notNull(),
    metricValue: real('metric_value').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [primaryKey({ columns: [table.trainingRunId, table.epoch, table.split, table.metricName] })],
)

/* ------------------------------------------------------------------ */
/*  Run Evaluations                                                   */
/*                                                                     */
/*  One row per run's post-training evaluation report (confusion       */
/*  matrix / per-class stats / misclassified rows for classification, */
/*  MAE/RMSE/R² for regression) — see ai_service/services/evaluate.py */
/*  and the 'evaluation' RunEvent case in lib/microservice.ts. `report`*/
/*  is the whole bounded document (ai_service caps topErrors and skips*/
/*  the confusion matrix past ~200 classes) — always read whole, so   */
/*  normalizing it into columns would only add joins for no benefit.  */
/*  `accuracy`/`macroF1` are denormalized out of it purely so a       */
/*  run-comparison query can ORDER BY / filter without parsing jsonb. */
/* ------------------------------------------------------------------ */
export const evaluationSplitEnum = pgEnum('evaluation_split', EvaluationSplits)
export const evaluationStatusEnum = pgEnum('evaluation_status', EvaluationStatuses)

export const runEvaluations = pgTable('run_evaluations', {
  // A run is evaluated at most once (the worker picks exactly one split via
  // its test -> validation -> full fallback ladder — see
  // ai_service/services/evaluate.py), so `runId` alone is both the natural
  // key and the primary key: a redelivered 'evaluation' event upserts this
  // row in place instead of accumulating duplicates.
  runId: uuid('run_id')
    .primaryKey()
    .references(() => trainingRuns.id, { onDelete: 'cascade' }),
  status: evaluationStatusEnum('status').notNull(),
  split: evaluationSplitEnum('split'),
  // S3 keys (BUCKET_TRAINING) — report is ingested into `report` below;
  // predictions is full-per-row and download-only, never read by the gateway.
  reportKey: text('report_key'),
  predictionsKey: text('predictions_key'),
  report: jsonb('report'),
  // Denormalized from `report` for cheap sorting/filtering (run comparison,
  // leaderboards) without parsing jsonb on every query.
  accuracy: real('accuracy'),
  macroF1: real('macro_f1'),
  failedMessage: text('failed_message'),
  evaluatedAt: timestamp('evaluated_at', { withTimezone: true }).defaultNow().notNull(),
})

/* ------------------------------------------------------------------ */
/*  Exports                                                           */
/*                                                                     */
/*  One row per requested export bundle. `model` tier just needs the  */
/*  converted artifact (from `theseus-models/{runId}/model.{format}`, */
/*  produced by the existing NATS export task); `devkit`/`app` also   */
/*  need the gateway to assemble a zip (see lib/export/*). `status`   */
/*  tracks that two-phase flow: pending -> converting (waiting on the */
/*  worker) -> assembling (gateway building the zip) -> ready|failed. */
/* ------------------------------------------------------------------ */
export const exportTierEnum = pgEnum('export_tier', ExportTiers)
export const exportFormatEnum = pgEnum('export_format', ExportFormats)
export const exportLangEnum = pgEnum('export_lang', ExportLangs)
export const exportStatusEnum = pgEnum('export_status', ExportStatuses)

// Named `modelExports` (not `exports`, a reserved-sounding identifier) —
// the pgTable's actual table name is still 'exports'.
export const modelExports = pgTable(
  'exports',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    runId: uuid('run_id')
      .notNull()
      .references(() => trainingRuns.id, { onDelete: 'cascade' }),
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    tier: exportTierEnum('tier').notNull(),
    format: exportFormatEnum('format').notNull(),
    // NULL for tier: 'model' — only devkit/app generate a client library.
    lang: exportLangEnum('lang'),
    status: exportStatusEnum('status').default('pending').notNull(),
    // The dispatched NATS export-task job id; lets the run-event consumer
    // find this row via `WHERE conversion_job_id = $jobId` when the worker
    // finishes converting the model artifact.
    conversionJobId: uuid('conversion_job_id'),
    // S3 key of the assembled zip, once ready.
    bundleKey: text('bundle_key'),
    byteSize: integer('byte_size'),
    checksum: char('checksum', { length: 64 }),
    failedMessage: text('failed_message'),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    readyAt: timestamp('ready_at', { withTimezone: true }),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [
    index('exports_runId_idx').on(table.runId),
    index('exports_conversionJobId_idx').on(table.conversionJobId),
    index('exports_status_idx').on(table.status),
  ],
)

/* ------------------------------------------------------------------ */
/*  Inference Jobs                                                    */
/*                                                                     */
/*  One row per dispatched inference job, written 'pending' at         */
/*  POST /inference/:runId and updated to success/failed by a durable  */
/*  gateway consumer on THESEUS_INFERENCE_RESULTS (see                */
/*  lib/microservice.ts's startInferenceResultsConsumer) the moment    */
/*  the worker publishes a terminal result — not lazily on poll, so a  */
/*  result is never lost even if nobody polls before that stream's     */
/*  1-hour retention window expires (see README's NATS subject table).*/
/*  `id` is the inferenceId chosen at dispatch time, not a fresh       */
/*  uuidv7 — the consumer has nothing else to key its update on, since */
/*  InferenceResponseSchema's payload never carries the inferenceId    */
/*  (only the subject's trailing token does).                         */
/* ------------------------------------------------------------------ */
export const inferenceJobStatusEnum = pgEnum('inference_job_status', InferenceJobStatuses)

export const inferenceJobs = pgTable(
  'inference_jobs',
  {
    id: uuid('id').primaryKey(),
    runId: uuid('run_id')
      .notNull()
      .references(() => trainingRuns.id, { onDelete: 'cascade' }),
    status: inferenceJobStatusEnum('status').default('pending').notNull(),
    // InferenceOutput on success — see server/lib/nats.ts's InferenceOutput union.
    output: jsonb('output'),
    error: text('error'),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    completedAt: timestamp('completed_at', { withTimezone: true }),
  },
  (table) => [
    index('inferenceJobs_runId_idx').on(table.runId),
    index('inferenceJobs_status_idx').on(table.status),
  ],
)

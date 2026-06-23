import { AnnotationTypes, AudioCodecs, DatasetModalities, ImageFormats, ProjectTasks, SplitTypes, TrainingStatuses } from '@server/lib/enums'
import { sql } from 'drizzle-orm'
import {
  boolean,
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

// --- 1. DATASET REGISTRY ---

export const datasets = pgTable('datasets', {
  projectId: uuid('project_id')
    .notNull()
    .primaryKey()
    .references(() => projects.id, { onDelete: 'cascade' }),
  // name: varchar('name', { length: 255 }).notNull(),
  // description: text('description'),
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

    // Only NOT NULL when modality is vision and is a snapshot version
    augmentationConfig: jsonb('augmentation_config'),

    // If NULL => draft dataset (project's working copy)
    // If NOT NULL => immutable snapshot (used for training runs)
    versionTag: varchar('version_tag', { length: 50 }),

    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index('datasetVersions_datasetId_idx').on(table.datasetId),
    unique('datasetVersions_datasetId_versionTag_key').on(table.datasetId, table.versionTag),
  ],
)

export const datasetSplits = pgTable(
  'dataset_splits',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    datasetVersionId: uuid('dataset_version_id')
      .notNull()
      .references(() => datasetVersions.id, { onDelete: 'cascade' }),
    splitType: splitTypeEnum('split_type').notNull(),
  },
  (table) => [
    index('datasetSplits_datasetVersionId_idx').on(table.datasetVersionId),
    unique('datasetSplits_datasetVersionId_splitType_key').on(table.datasetVersionId, table.splitType),
  ],
)

// --- 2. CENTRAL DATA ITEM TABLE ---

export const datasetItems = pgTable(
  'dataset_items',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    datasetSplitId: uuid('dataset_split_id')
      .notNull()
      .references(() => datasetSplits.id, { onDelete: 'cascade' }),
    externalId: varchar('external_id', { length: 255 }), // ID mapping back to source cloud storage or local path
    storageUrl: text('storage_url'), // Path to raw files (S3, GCS) if unstructured (images/audio/raw text files)
    // embedding: vector('embedding', { dimensions: 1536 }), // Vector representation of the data item for semantic filtering/filtering out duplicates
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index('dataset_items_datasetSplitId_idx').on(table.datasetSplitId),
    // TODO: Add embedding index when we decide to use it
    // index('dataset_items_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops')),
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

// E. Reinforcement Learning Modality (RLHF, PPO, DPO Trajectories)
// export const rlTrajectories = pgTable('rl_trajectories', {
//   itemId: uuid('item_id')
//     .primaryKey()
//     .references(() => datasetItems.id, { onDelete: 'cascade' }),
//   stateRepresentation: jsonb('state_representation').notNull(),
//   actionTaken: jsonb('action_taken').notNull(),
//   nextStateRepresentation: jsonb('next_state_representation'),
//   reward: numeric('reward', { precision: 12, scale: 6 }),
//   isTerminal: boolean('is_terminal').default(false),
//   timestepIndex: integer('timestep_index').notNull(),
// })

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
  ],
)

export const annotations = pgTable(
  'annotations',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    itemId: uuid('item_id').references(() => datasetItems.id, { onDelete: 'cascade' }),
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
    // Drizzle table-level check constraint for confidence limits
    check('confidence_bounds', sql`${table.confidenceScore} BETWEEN 0.0 AND 1.0`),
  ],
)

// export const datasetClasses = pgTable(
//   'dataset_classes',
//   {
//     id: uuid('id').primaryKey().default(sql`uuidv7()`),
//     projectId: uuid('project_id')
//       .notNull()
//       .references(() => projects.id, { onDelete: 'cascade' }),
//     name: text('name').notNull(),
//     color: text('color').notNull().default('#e03131'),
//   },
//   (table) => [
//     index('datasetClasses_projectId_idx').on(table.projectId),
//     unique('datasetClasses_projectId_name_key').on(table.projectId, table.name),
//   ],
// )

// export const splitEnum = pgEnum('dataset_split', ['train', 'validation', 'test'])

/* ------------------------------------------------------------------ */
/*  Dataset Images                                                    */
/* ------------------------------------------------------------------ */
// export const datasetImages = pgTable(
//   'dataset_images',
//   {
//     id: uuid('id').primaryKey().default(sql`uuidv7()`),
//     projectId: uuid('project_id')
//       .notNull()
//       .references(() => projects.id, { onDelete: 'cascade' }),
//     classId: uuid('class_id').references(() => datasetClasses.id, { onDelete: 'set null' }),
//     filename: text('filename').notNull(),
//     path: text('path').notNull(),
//     width: integer('width').notNull(),
//     height: integer('height').notNull(),
//     split: splitEnum(),
//     uploadedAt: timestamp('uploaded_at', { withTimezone: true }).defaultNow().notNull(),
//   },
//   (table) => [
//     index('datasetImages_projectId_idx').on(table.projectId),
//     index('datasetImages_classId_idx').on(table.classId),
//   ],
// )

// Status explained:
// queued: queued for training, waiting for free GPU
// training: training
// completed: training completed (stopped = completed + completedAt is null, failed = completed + failedMessage is not null)
export const trainingStatusEnum = pgEnum('training_status', TrainingStatuses)

/* ------------------------------------------------------------------ */
/*  Training Runs                                                     */
/* ------------------------------------------------------------------ */
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

    // Configuration (Using JSONB for flexible hyperparams)
    hyperparameters: jsonb('hyperparameters').notNull(),

    startedAt: timestamp('started_at', { withTimezone: true }),
    completedAt: timestamp('completed_at', { withTimezone: true }),
    failedMessage: text('failed_message'),

    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [index('trainingRuns_projectId_idx').on(table.projectId)],
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
    trainingLoss: real('training_loss').notNull(),
    validationLoss: real('validation_loss').notNull(),
    accuracy: real('accuracy').notNull(),
    mAP: real('mean_average_precision').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [primaryKey({ columns: [table.trainingRunId, table.epoch] })],
)

// export const trainingMetrics = pgTable(
//   'training_metrics',
//   {
//     trainingRunId: uuid('training_run_id')
//       .notNull()
//       .references(() => trainingRuns.id, { onDelete: 'cascade' }),
//     epoch: integer('epoch').notNull(),
//     metricName: varchar('metric_name', { length: 255 }).notNull(),
//     metricValue: numeric('metric_value', { precision: 12, scale: 6 }).notNull(),
//     createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
//   },
//   (table) => [primaryKey({ columns: [table.trainingRunId, table.epoch, table.metricName] })],
// )

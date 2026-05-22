import { sql } from 'drizzle-orm'
import {
  boolean,
  check,
  index,
  integer,
  pgEnum,
  pgTable,
  primaryKey,
  real,
  text,
  timestamp,
  unique,
  uuid,
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
export const projects = pgTable(
  'projects',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    name: text('name').notNull(),
    description: text('description').default(''),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [index('projects_userId_idx').on(table.userId)],
)

/* ------------------------------------------------------------------ */
/*  Dataset Classes                                                   */
/* ------------------------------------------------------------------ */
export const datasetClasses = pgTable(
  'dataset_classes',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    projectId: uuid('project_id')
      .notNull()
      .references(() => projects.id, { onDelete: 'cascade' }),
    name: text('name').notNull(),
    color: text('color').notNull().default('#e03131'),
  },
  (table) => [
    index('datasetClasses_projectId_idx').on(table.projectId),
    unique('datasetClasses_projectId_name_key').on(table.projectId, table.name),
  ],
)

/* ------------------------------------------------------------------ */
/*  Dataset Images                                                    */
/* ------------------------------------------------------------------ */
export const datasetImages = pgTable(
  'dataset_images',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    projectId: uuid('project_id')
      .notNull()
      .references(() => projects.id, { onDelete: 'cascade' }),
    classId: uuid('class_id')
      .references(() => datasetClasses.id, { onDelete: 'set null' }),
    filename: text('filename').notNull(),
    path: text('path').notNull(),
    width: integer('width').notNull(),
    height: integer('height').notNull(),
    uploadedAt: timestamp('uploaded_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [
    index('datasetImages_projectId_idx').on(table.projectId),
    index('datasetImages_classId_idx').on(table.classId),
  ],
)

/* ------------------------------------------------------------------ */
/*  Bounding Box Labels                                               */
/* ------------------------------------------------------------------ */
// export const labels = pgTable(
//   'labels',
//   {
//     id: uuid('id').primaryKey().default(sql`uuidv7()`),
//     imageId: uuid('image_id')
//       .notNull()
//       .references(() => datasetImages.id, { onDelete: 'cascade' }),
//     classId: uuid('class_id')
//       .notNull()
//       .references(() => datasetClasses.id, { onDelete: 'cascade' }),
//     // Note: the coordinates are normalized [0, 1], x and y are the center of the bounding box (YOLO format)
//     x: real('x').notNull(),
//     y: real('y').notNull(),
//     width: real('width').notNull(),
//     height: real('height').notNull(),
//   },
//   (table) => [index('labels_imageId_idx').on(table.imageId), index('labels_classId_idx').on(table.classId)],
// )

export const trainingStatusEnum = pgEnum('training_status', ['queued', 'training', 'completed', 'failed'])

/* ------------------------------------------------------------------ */
/*  Training Runs                                                     */
/* ------------------------------------------------------------------ */
export const trainingRuns = pgTable(
  'training_runs',
  {
    id: uuid('id').primaryKey().default(sql`uuidv7()`),
    projectId: uuid('project_id')
      .notNull()
      .references(() => projects.id, { onDelete: 'cascade' }),
    taskId: text('task_id'),
    modelId: text('model_id').notNull(),
    variantId: text('variant_id').notNull(),
    train: integer('train').notNull().default(70),
    validate: integer('validate').notNull().default(20),
    test: integer('test').notNull().default(10),
    epochs: integer('epochs').notNull().default(50),
    batchSize: integer('batch_size').notNull().default(16),
    learningRate: real('learning_rate').notNull().default(0.001),
    imageSize: integer('image_size').notNull().default(224),
    status: trainingStatusEnum().default('queued').notNull(),
    startedAt: timestamp('started_at', { withTimezone: true }),
    completedAt: timestamp('completed_at', { withTimezone: true }),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true })
      .$onUpdate(() => new Date())
      .notNull(),
  },
  (table) => [
    index('trainingRuns_projectId_idx').on(table.projectId),
    check('split_config_check', sql`${table.train} + ${table.validate} + ${table.test} = 100`),
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
    trainingLoss: real('training_loss').notNull(),
    validationLoss: real('validation_loss').notNull(),
    accuracy: real('accuracy').notNull(),
    mAP: real('mean_average_precision').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [primaryKey({ columns: [table.trainingRunId, table.epoch] })],
)

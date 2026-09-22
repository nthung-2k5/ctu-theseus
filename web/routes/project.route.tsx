import { projectDetailQueryOptions } from '@public/lib/queries'
import { ClassesPage } from '@public/pages/ClassesPage'
import { DatasetPage } from '@public/pages/DatasetPage'
import { ProjectPage } from '@public/pages/ProjectPage'
import { SnapshotsPage } from '@public/pages/SnapshotsPage'
import { TrainingPage } from '@public/pages/TrainingPage'
import { UploadPage } from '@public/pages/UploadPage'
import { createRoute, Outlet, stripSearchParams } from '@tanstack/react-router'
import { z } from 'zod'
import { appRoute } from './app.route'

/** Loads (and caches) the project once for every child route — the workflow steps, nav gating, and page bodies all read the same query. */
export const projectRoute = createRoute({
  getParentRoute: () => appRoute,
  path: 'project/$projectId',
  loader: ({ context, params }) => context.queryClient.ensureQueryData(projectDetailQueryOptions(params.projectId)),
  component: () => <Outlet />,
})

export const projectIndexRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/',
  component: ProjectPage,
})

export const uploadRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/upload',
  component: UploadPage,
})

export const classesRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/classes',
  component: ClassesPage,
})

const paginationDefaults = { page: 1, perPage: 20 }

const itemSortSchema = z.enum(['newest', 'oldest', 'filename']).optional().catch(undefined)

// classId also carries the 'unassigned' sentinel (items with no
// classification annotation), so it isn't a plain uuid.
const itemClassIdSchema = z
  .union([z.literal('unassigned'), z.uuid()])
  .optional()
  .catch(undefined)

// The draft is a single, implicit version — no versionId to pick.
const datasetSearchSchema = z.object({
  page: z.coerce.number().int().min(1).catch(paginationDefaults.page),
  perPage: z.coerce.number().int().min(1).max(1000).catch(paginationDefaults.perPage),
  split: z.enum(['train', 'validation', 'test']).optional().catch(undefined),
  classId: itemClassIdSchema,
  search: z.string().optional().catch(undefined),
  sort: itemSortSchema,
})

export const datasetRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/dataset',
  validateSearch: datasetSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: DatasetPage,
})

const snapshotsSearchSchema = z.object({
  page: z.coerce.number().int().min(1).catch(paginationDefaults.page),
  perPage: z.coerce.number().int().min(1).max(1000).catch(paginationDefaults.perPage),
  versionId: z.uuid().optional().catch(undefined),
  split: z.enum(['train', 'validation', 'test']).optional().catch(undefined),
  classId: itemClassIdSchema,
  search: z.string().optional().catch(undefined),
  sort: itemSortSchema,
  // Absent means every item. Only meaningful for a snapshot built with augmentation.
  origin: z.enum(['original', 'augmented']).optional().catch(undefined),
})

export const snapshotsRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/snapshots',
  validateSearch: snapshotsSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: SnapshotsPage,
})

/**
 * `tab` picks which pane of the selected run to show. Export and inference
 * used to be their own routes (/models, /inference); they're now tabs beside
 * the run's metrics, so a run and the thing you want to do with it live in
 * one URL. Absent means Metrics.
 */
const trainingSearchSchema = z.object({
  runId: z.string().optional().catch(undefined),
  tab: z.enum(['metrics', 'evaluation', 'export', 'inference']).optional().catch(undefined),
  // Set when the user opens the "New Run"/"New Sweep" page from the run
  // list; cleared once a run/sweep is selected.
  view: z.enum(['create', 'sweep']).optional().catch(undefined),
  // Set when viewing one sweep's trial leaderboard.
  sweepId: z.string().optional().catch(undefined),
})

export const trainingRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/training',
  validateSearch: trainingSearchSchema.parse,
  component: TrainingPage,
})

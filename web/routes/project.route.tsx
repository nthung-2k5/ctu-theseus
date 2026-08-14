import { projectDetailQueryOptions } from '@public/lib/queries'
import { ClassesPage } from '@public/pages/ClassesPage'
import { DataPage } from '@public/pages/DataPage'
import { DatasetPage } from '@public/pages/DatasetPage'
import { InferencePage } from '@public/pages/InferencePage'
import { ModelsPage } from '@public/pages/ModelsPage'
import { ProjectPage } from '@public/pages/ProjectPage'
import { TrainingPage } from '@public/pages/TrainingPage'
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

const paginationDefaults = { page: 1, perPage: 20 }
const paginationSearchSchema = z.object({
  page: z.coerce.number().int().min(1).catch(paginationDefaults.page),
  perPage: z.coerce.number().int().min(1).max(200).catch(paginationDefaults.perPage),
  versionId: z.uuid().optional().catch(undefined),
  split: z.enum(['train', 'validation', 'test']).optional().catch(undefined),
})

export const dataRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/data',
  validateSearch: paginationSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: DataPage,
})

export const datasetRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/dataset',
  validateSearch: paginationSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: DatasetPage,
})

export const classesRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/classes',
  component: ClassesPage,
})

const runSelectionSchema = z.object({
  runId: z.string().optional().catch(undefined),
})

export const trainingRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/training',
  validateSearch: runSelectionSchema.parse,
  component: TrainingPage,
})

export const modelsRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/models',
  validateSearch: runSelectionSchema.parse,
  component: ModelsPage,
})

export const inferenceRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/inference',
  validateSearch: runSelectionSchema.parse,
  component: InferencePage,
})

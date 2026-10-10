import { projectDetailQueryOptions } from '@public/lib/queries'
import { DataPage } from '@public/pages/DataPage'
import { ProjectPage } from '@public/pages/ProjectPage'
import { SnapshotBuilderPage } from '@public/pages/SnapshotBuilderPage'
import { SnapshotDetailPage } from '@public/pages/SnapshotDetailPage'
import { SnapshotsPage } from '@public/pages/SnapshotsPage'
import { NewTrainingPage } from '@public/pages/training/NewTrainingPage'
import { RunComparePage } from '@public/pages/training/RunComparePage'
import { RunConfigPage } from '@public/pages/training/RunConfigPage'
import { RunEvaluationPage } from '@public/pages/training/RunEvaluationPage'
import { RunExportPage } from '@public/pages/training/RunExportPage'
import { RunLayout } from '@public/pages/training/RunLayout'
import { RunLivePage } from '@public/pages/training/RunLivePage'
import { RunTryPage } from '@public/pages/training/RunTryPage'
import { SweepPage } from '@public/pages/training/SweepPage'
import { TrainingPage } from '@public/pages/training/TrainingPage'
import { createRoute, Outlet, redirect, stripSearchParams } from '@tanstack/react-router'
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

const itemSortSchema = z.enum(['newest', 'oldest', 'filename']).optional().catch(undefined)

// Only for tasks whose ground truth isn't a class (captioning, regression, ...): class tasks use
// classId=unassigned instead. Absent means no filtering.
const itemLabeledSchema = z.enum(['labeled', 'unlabeled']).optional().catch(undefined)

// classId also carries the 'unassigned' sentinel (items with no
// classification annotation), so it isn't a plain uuid.
const itemClassIdSchema = z
  .union([z.literal('unassigned'), z.uuid()])
  .optional()
  .catch(undefined)

// The draft is a single, implicit version — no versionId to pick.
const dataSearchSchema = z.object({
  page: z.coerce.number().int().min(1).catch(paginationDefaults.page),
  perPage: z.coerce.number().int().min(1).max(1000).catch(paginationDefaults.perPage),
  split: z.enum(['train', 'validation', 'test']).optional().catch(undefined),
  classId: itemClassIdSchema,
  search: z.string().optional().catch(undefined),
  sort: itemSortSchema,
  labeled: itemLabeledSchema,
  // Absent means Browse. Classes only exists for tasks that label with classes.
  tab: z.enum(['browse', 'classes', 'distribution']).optional().catch(undefined),
  // The add-data drawer, so the overview and the empty states can open it with a link.
  add: z.literal(true).optional().catch(undefined),
})

/** The project's data in one place: browse the draft, manage its classes, check its health, add more. */
export const dataRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/data',
  validateSearch: dataSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: DataPage,
})

/* Old URLs: the dataset, upload and classes pages are tabs and a drawer of the data page now. */
export const legacyDatasetRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/dataset',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/data', params, search: { ...paginationDefaults } })
  },
})
export const legacyUploadRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/upload',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/data', params, search: { ...paginationDefaults, add: true } })
  },
})
export const legacyClassesRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/classes',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/data', params, search: { ...paginationDefaults, tab: 'classes' } })
  },
})

/** Search params for browsing one snapshot's items. */
const snapshotItemsSearchSchema = z.object({
  page: z.coerce.number().int().min(1).catch(paginationDefaults.page),
  perPage: z.coerce.number().int().min(1).max(1000).catch(paginationDefaults.perPage),
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
  component: () => <Outlet />,
})

export const snapshotsIndexRoute = createRoute({
  getParentRoute: () => snapshotsRoute,
  path: '/',
  component: SnapshotsPage,
})

export const snapshotNewRoute = createRoute({
  getParentRoute: () => snapshotsRoute,
  path: '/new',
  component: SnapshotBuilderPage,
})

export const snapshotDetailRoute = createRoute({
  getParentRoute: () => snapshotsRoute,
  path: '/$versionId',
  validateSearch: snapshotItemsSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: SnapshotDetailPage,
})

/* ── Training (runs and sweeps) ── */

export const trainingRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/training',
  component: () => <Outlet />,
})

/**
 * The runs list. Training used to be one page at this same path, switched by search params (`runId`, `view`,
 * `sweepId`, `tab`); these now live at their own paths, so old links and bookmarks carrying them are
 * redirected. With none of them this is the list itself.
 */
export const trainingIndexRoute = createRoute({
  getParentRoute: () => trainingRoute,
  path: '/',
  // Optional keys, so a plain link to the list needs no `search`. `runId`, `view`, `sweepId` and `tab` are the old
  // single-page URLs, redirected below; `q`, `status`, `sort` and `show` are the list's own filters.
  validateSearch: z.object({
    runId: z.string().optional().catch(undefined),
    view: z.enum(['create', 'sweep']).optional().catch(undefined),
    sweepId: z.string().optional().catch(undefined),
    tab: z.string().optional().catch(undefined),
    q: z.string().optional().catch(undefined),
    // Comma-separated run statuses.
    status: z.string().optional().catch(undefined),
    sort: z.enum(['newest', 'best']).optional().catch(undefined),
    show: z.enum(['runs', 'sweeps']).optional().catch(undefined),
  }).parse,
  beforeLoad: ({ params, search }) => {
    const { projectId } = params
    if (search.sweepId)
      throw redirect({
        to: '/project/$projectId/training/sweeps/$sweepId',
        params: { projectId, sweepId: search.sweepId },
      })
    if (search.view)
      throw redirect({
        to: '/project/$projectId/training/new',
        params: { projectId },
        search: { mode: search.view === 'sweep' ? 'sweep' : 'run' },
      })
    if (search.runId) {
      const runId = search.runId
      if (search.tab === 'evaluation')
        throw redirect({ to: '/project/$projectId/training/$runId/evaluation', params: { projectId, runId } })
      if (search.tab === 'inference')
        throw redirect({ to: '/project/$projectId/training/$runId/try', params: { projectId, runId } })
      if (search.tab === 'export')
        throw redirect({ to: '/project/$projectId/training/$runId/export', params: { projectId, runId } })
      throw redirect({ to: '/project/$projectId/training/$runId', params: { projectId, runId } })
    }
  },
  component: TrainingPage,
})

export const trainingNewRoute = createRoute({
  getParentRoute: () => trainingRoute,
  path: '/new',
  // `from` pre-fills the form from an earlier run ("New run from this setup").
  validateSearch: z.object({
    mode: z.enum(['run', 'sweep']).optional().catch(undefined),
    from: z.string().optional().catch(undefined),
  }).parse,
  component: NewTrainingPage,
})

export const sweepRoute = createRoute({
  getParentRoute: () => trainingRoute,
  path: '/sweeps/$sweepId',
  component: SweepPage,
})

/** Any number of runs side by side. The list's selection bar links here; a run's own page can start one too. */
export const compareRoute = createRoute({
  getParentRoute: () => trainingRoute,
  path: '/compare',
  // Comma-separated run ids.
  validateSearch: z.object({ runs: z.string().optional().catch(undefined) }).parse,
  component: RunComparePage,
})

/** The run layout owns the SSE stream and tab strip; each tab is a child route. */
export const runRoute = createRoute({
  getParentRoute: () => trainingRoute,
  path: '/$runId',
  component: RunLayout,
})

export const runLiveRoute = createRoute({ getParentRoute: () => runRoute, path: '/', component: RunLivePage })
export const runEvaluationRoute = createRoute({
  getParentRoute: () => runRoute,
  path: '/evaluation',
  component: RunEvaluationPage,
})
export const runTryRoute = createRoute({ getParentRoute: () => runRoute, path: '/try', component: RunTryPage })
export const runExportRoute = createRoute({ getParentRoute: () => runRoute, path: '/export', component: RunExportPage })
export const runConfigRoute = createRoute({ getParentRoute: () => runRoute, path: '/config', component: RunConfigPage })

/** Old URL: comparing is a page of its own now, which this run is the first entry of. */
export const runCompareRedirectRoute = createRoute({
  getParentRoute: () => runRoute,
  path: '/compare',
  validateSearch: z.object({ with: z.string().optional().catch(undefined) }).parse,
  beforeLoad: ({ params, search }) => {
    const ids = [params.runId, ...(search.with ?? '').split(',').filter((id) => id && id !== params.runId)]
    throw redirect({ to: '/project/$projectId/training/compare', params, search: { runs: ids.join(',') } })
  },
})

/* ── Old playground and export URLs: both are tabs of the run now ── */

export const legacyPlaygroundRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/playground',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/training', params })
  },
})
export const legacyPlaygroundRunRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/playground/$runId',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/training/$runId/try', params })
  },
})
export const legacyExportRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/export',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/training', params })
  },
})
export const legacyExportRunRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/export/$runId',
  beforeLoad: ({ params }) => {
    throw redirect({ to: '/project/$projectId/training/$runId/export', params })
  },
})

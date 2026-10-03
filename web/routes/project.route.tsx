import { projectDetailQueryOptions } from '@public/lib/queries'
import { ClassesPage } from '@public/pages/ClassesPage'
import { DatasetPage } from '@public/pages/DatasetPage'
import { ExportPage } from '@public/pages/export/ExportPage'
import { ExportRunPage } from '@public/pages/export/ExportRunPage'
import { ProjectPage } from '@public/pages/ProjectPage'
import { PlaygroundPage } from '@public/pages/playground/PlaygroundPage'
import { PlaygroundRunPage } from '@public/pages/playground/PlaygroundRunPage'
import { SnapshotBuilderPage } from '@public/pages/SnapshotBuilderPage'
import { SnapshotDetailPage } from '@public/pages/SnapshotDetailPage'
import { SnapshotsPage } from '@public/pages/SnapshotsPage'
import { NewTrainingPage } from '@public/pages/training/NewTrainingPage'
import { RunComparePage } from '@public/pages/training/RunComparePage'
import { RunConfigPage } from '@public/pages/training/RunConfigPage'
import { RunEvaluationPage } from '@public/pages/training/RunEvaluationPage'
import { RunLayout } from '@public/pages/training/RunLayout'
import { RunLivePage } from '@public/pages/training/RunLivePage'
import { SweepPage } from '@public/pages/training/SweepPage'
import { TrainingPage } from '@public/pages/training/TrainingPage'
import { UploadPage } from '@public/pages/UploadPage'
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
  // Absent means Browse.
  tab: z.enum(['browse', 'distribution']).optional().catch(undefined),
})

export const datasetRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/dataset',
  validateSearch: datasetSearchSchema.parse,
  search: { middlewares: [stripSearchParams(paginationDefaults)] },
  component: DatasetPage,
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
  // Optional keys, so a plain link to the list needs no `search`.
  validateSearch: z.object({
    runId: z.string().optional().catch(undefined),
    view: z.enum(['create', 'sweep']).optional().catch(undefined),
    sweepId: z.string().optional().catch(undefined),
    tab: z.string().optional().catch(undefined),
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
        throw redirect({ to: '/project/$projectId/playground/$runId', params: { projectId, runId } })
      if (search.tab === 'export')
        throw redirect({ to: '/project/$projectId/export/$runId', params: { projectId, runId } })
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
export const runCompareRoute = createRoute({
  getParentRoute: () => runRoute,
  path: '/compare',
  // Comma-separated run ids to compare against; the current run is always included.
  validateSearch: z.object({ with: z.string().optional().catch(undefined) }).parse,
  component: RunComparePage,
})
export const runConfigRoute = createRoute({ getParentRoute: () => runRoute, path: '/config', component: RunConfigPage })

/* ── Playground and export ── */

export const playgroundRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/playground',
  component: () => <Outlet />,
})
export const playgroundIndexRoute = createRoute({
  getParentRoute: () => playgroundRoute,
  path: '/',
  component: PlaygroundPage,
})
export const playgroundRunRoute = createRoute({
  getParentRoute: () => playgroundRoute,
  path: '/$runId',
  component: PlaygroundRunPage,
})

export const exportRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/export',
  component: () => <Outlet />,
})
export const exportIndexRoute = createRoute({ getParentRoute: () => exportRoute, path: '/', component: ExportPage })
export const exportRunRoute = createRoute({
  getParentRoute: () => exportRoute,
  path: '/$runId',
  component: ExportRunPage,
})

import { rootRoute } from './__root'
import {
  adminIndexRoute,
  adminJobsRoute,
  adminModelsRoute,
  adminPluginsRoute,
  adminRoute,
  adminUsersRoute,
} from './admin.route'
import { apiKeysRoute, appRoute, dashboardRoute, importedModelsRoute, settingsRoute } from './app.route'
import { guestRoute, landingRoute, loginRoute, registerRoute } from './guest.route'
import {
  compareRoute,
  dataRoute,
  legacyClassesRoute,
  legacyDatasetRoute,
  legacyExportRoute,
  legacyExportRunRoute,
  legacyPlaygroundRoute,
  legacyPlaygroundRunRoute,
  legacyUploadRoute,
  projectIndexRoute,
  projectRoute,
  runCompareRedirectRoute,
  runConfigRoute,
  runEvaluationRoute,
  runExportRoute,
  runLiveRoute,
  runRoute,
  runTryRoute,
  snapshotDetailRoute,
  snapshotNewRoute,
  snapshotsIndexRoute,
  snapshotsRoute,
  sweepRoute,
  trainingIndexRoute,
  trainingNewRoute,
  trainingRoute,
} from './project.route'

const guestTree = guestRoute.addChildren([loginRoute])

const projectTree = projectRoute.addChildren([
  projectIndexRoute,
  dataRoute,
  snapshotsRoute.addChildren([snapshotsIndexRoute, snapshotNewRoute, snapshotDetailRoute]),
  trainingRoute.addChildren([
    trainingIndexRoute,
    trainingNewRoute,
    sweepRoute,
    compareRoute,
    runRoute.addChildren([
      runLiveRoute,
      runEvaluationRoute,
      runTryRoute,
      runExportRoute,
      runConfigRoute,
      runCompareRedirectRoute,
    ]),
  ]),
  legacyDatasetRoute,
  legacyUploadRoute,
  legacyClassesRoute,
  legacyPlaygroundRoute,
  legacyPlaygroundRunRoute,
  legacyExportRoute,
  legacyExportRunRoute,
])

const adminTree = adminRoute.addChildren([
  adminIndexRoute,
  adminUsersRoute,
  adminModelsRoute,
  adminPluginsRoute,
  adminJobsRoute,
])

const appTree = appRoute.addChildren([
  dashboardRoute,
  importedModelsRoute,
  settingsRoute,
  apiKeysRoute,
  adminTree,
  projectTree,
])

export const routeTree = rootRoute.addChildren([landingRoute, registerRoute, guestTree, appTree])

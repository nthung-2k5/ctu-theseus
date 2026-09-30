import { rootRoute } from './__root'
import {
  adminIndexRoute,
  adminJobsRoute,
  adminModelsRoute,
  adminPluginsRoute,
  adminRoute,
  adminUsersRoute,
} from './admin.route'
import { apiKeysRoute, appRoute, dashboardRoute, modelsRoute, settingsRoute } from './app.route'
import { guestRoute, landingRoute, loginRoute, registerRoute } from './guest.route'
import {
  classesRoute,
  datasetRoute,
  experimentNewRoute,
  experimentsIndexRoute,
  experimentsRoute,
  exportIndexRoute,
  exportRoute,
  exportRunRoute,
  legacyTrainingRoute,
  playgroundIndexRoute,
  playgroundRoute,
  playgroundRunRoute,
  projectIndexRoute,
  projectRoute,
  runCompareRoute,
  runConfigRoute,
  runEvaluationRoute,
  runLiveRoute,
  runRoute,
  snapshotDetailRoute,
  snapshotNewRoute,
  snapshotsIndexRoute,
  snapshotsRoute,
  sweepRoute,
  uploadRoute,
} from './project.route'

const guestTree = guestRoute.addChildren([loginRoute])

const projectTree = projectRoute.addChildren([
  projectIndexRoute,
  uploadRoute,
  classesRoute,
  datasetRoute,
  snapshotsRoute.addChildren([snapshotsIndexRoute, snapshotNewRoute, snapshotDetailRoute]),
  experimentsRoute.addChildren([
    experimentsIndexRoute,
    experimentNewRoute,
    sweepRoute,
    runRoute.addChildren([runLiveRoute, runEvaluationRoute, runCompareRoute, runConfigRoute]),
  ]),
  playgroundRoute.addChildren([playgroundIndexRoute, playgroundRunRoute]),
  exportRoute.addChildren([exportIndexRoute, exportRunRoute]),
  legacyTrainingRoute,
])

const adminTree = adminRoute.addChildren([
  adminIndexRoute,
  adminUsersRoute,
  adminModelsRoute,
  adminPluginsRoute,
  adminJobsRoute,
])

const appTree = appRoute.addChildren([dashboardRoute, modelsRoute, settingsRoute, apiKeysRoute, adminTree, projectTree])

export const routeTree = rootRoute.addChildren([landingRoute, registerRoute, guestTree, appTree])

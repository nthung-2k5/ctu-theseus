import { rootRoute } from './__root'
import { apiKeysRoute, appRoute, dashboardRoute, settingsRoute } from './app.route'
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
  runLogsRoute,
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
    runRoute.addChildren([runLiveRoute, runLogsRoute, runEvaluationRoute, runCompareRoute, runConfigRoute]),
  ]),
  playgroundRoute.addChildren([playgroundIndexRoute, playgroundRunRoute]),
  exportRoute.addChildren([exportIndexRoute, exportRunRoute]),
  legacyTrainingRoute,
])

const appTree = appRoute.addChildren([dashboardRoute, settingsRoute, apiKeysRoute, projectTree])

export const routeTree = rootRoute.addChildren([landingRoute, registerRoute, guestTree, appTree])

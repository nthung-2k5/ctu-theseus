import { rootRoute } from './__root'
import { apiKeysRoute, appRoute, dashboardRoute } from './app.route'
import { guestRoute, loginRoute, registerRoute } from './guest.route'
import {
  classesRoute,
  datasetRoute,
  projectIndexRoute,
  projectRoute,
  snapshotsRoute,
  trainingRoute,
  uploadRoute,
} from './project.route'

const guestTree = guestRoute.addChildren([loginRoute, registerRoute])

const projectTree = projectRoute.addChildren([
  projectIndexRoute,
  uploadRoute,
  classesRoute,
  datasetRoute,
  snapshotsRoute,
  trainingRoute,
])

const appTree = appRoute.addChildren([dashboardRoute, apiKeysRoute, projectTree])

export const routeTree = rootRoute.addChildren([guestTree, appTree])

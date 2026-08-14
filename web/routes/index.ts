import { rootRoute } from './__root'
import { appRoute, dashboardRoute } from './app.route'
import { guestRoute, loginRoute, registerRoute } from './guest.route'
import {
  classesRoute,
  dataRoute,
  datasetRoute,
  inferenceRoute,
  modelsRoute,
  projectIndexRoute,
  projectRoute,
  trainingRoute,
} from './project.route'

const guestTree = guestRoute.addChildren([loginRoute, registerRoute])

const projectTree = projectRoute.addChildren([
  projectIndexRoute,
  dataRoute,
  datasetRoute,
  classesRoute,
  trainingRoute,
  modelsRoute,
  inferenceRoute,
])

const appTree = appRoute.addChildren([dashboardRoute, projectTree])

export const routeTree = rootRoute.addChildren([guestTree, appTree])

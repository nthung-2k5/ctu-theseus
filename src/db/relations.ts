import { defineRelations } from 'drizzle-orm'
import * as schema from './schema'

export const relations = defineRelations(schema, (r) => ({
  accounts: {
    users: r.one.users({
      from: r.accounts.userId,
      to: r.users.id,
    }),
  },
  users: {
    accounts: r.many.accounts(),
    projects: r.many.projects(),
    sessions: r.many.sessions(),
  },
  sessions: {
    users: r.one.users({
      from: r.sessions.userId,
      to: r.users.id,
    }),
  },
  datasetClasses: {
    projects: r.one.projects({
      from: r.datasetClasses.projectId,
      to: r.projects.id,
    }),
    datasetImages: r.many.datasetImages({
      from: r.datasetClasses.id.through(r.labels.classId),
      to: r.datasetImages.id.through(r.labels.imageId),
    }),
  },
  projects: {
    datasetClasses: r.many.datasetClasses(),
    datasets: r.many.datasets(),
    users: r.one.users({
      from: r.projects.userId,
      to: r.users.id,
    }),
  },
  datasetImages: {
    datasets: r.one.datasets({
      from: r.datasetImages.datasetId,
      to: r.datasets.id,
    }),
    datasetClasses: r.many.datasetClasses(),
  },
  datasets: {
    datasetImages: r.many.datasetImages(),
    projects: r.one.projects({
      from: r.datasets.projectId,
      to: r.projects.id,
    }),
    trainingRuns: r.many.trainingRuns(),
  },
  trainingMetrics: {
    trainingRuns: r.one.trainingRuns({
      from: r.trainingMetrics.trainingRunId,
      to: r.trainingRuns.id,
    }),
  },
  trainingRuns: {
    trainingMetrics: r.many.trainingMetrics(),
    datasets: r.one.datasets({
      from: r.trainingRuns.datasetId,
      to: r.datasets.id,
    }),
  },
}))

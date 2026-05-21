import { defineRelations } from 'drizzle-orm'
import * as schema from './schema'

export const relations = defineRelations(schema, (r) => ({
  accounts: {
    user: r.one.users({
      from: r.accounts.userId,
      to: r.users.id,
    }),
  },
  users: {
    accounts: r.many.accounts(),
    projects: r.many.projects(),
    sessions: r.many.sessions(),
  },
  datasetClasses: {
    project: r.one.projects({
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
    user: r.one.users({
      from: r.projects.userId,
      to: r.users.id,
    }),
  },
  datasetImages: {
    dataset: r.one.datasets({
      from: r.datasetImages.datasetId,
      to: r.datasets.id,
    }),
    datasetClasses: r.many.datasetClasses(),
  },
  datasets: {
    datasetImages: r.many.datasetImages(),
    project: r.one.projects({
      from: r.datasets.projectId,
      to: r.projects.id,
    }),
    trainingRuns: r.many.trainingRuns(),
  },
  sessions: {
    user: r.one.users({
      from: r.sessions.userId,
      to: r.users.id,
    }),
  },
  trainingMetrics: {
    trainingRun: r.one.trainingRuns({
      from: r.trainingMetrics.trainingRunId,
      to: r.trainingRuns.id,
    }),
  },
  trainingRuns: {
    trainingMetrics: r.many.trainingMetrics(),
    dataset: r.one.datasets({
      from: r.trainingRuns.datasetId,
      to: r.datasets.id,
    }),
  },
}))

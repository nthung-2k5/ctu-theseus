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
    datasetImages: r.many.datasetImages(),
    trainingRuns: r.many.trainingRuns(),
    users: r.one.users({
      from: r.projects.userId,
      to: r.users.id,
    }),
  },
  datasetImages: {
    projects: r.one.projects({
      from: r.datasetImages.projectId,
      to: r.projects.id,
    }),
    datasetClasses: r.many.datasetClasses(),
  },
  trainingMetrics: {
    trainingRuns: r.one.trainingRuns({
      from: r.trainingMetrics.trainingRunId,
      to: r.trainingRuns.id,
    }),
  },
  trainingRuns: {
    trainingMetrics: r.many.trainingMetrics(),
    projects: r.one.projects({
      from: r.trainingRuns.projectId,
      to: r.projects.id,
    }),
  },
}))

import { defineRelationsPart } from 'drizzle-orm'
import * as schema from './schema'

const authRelations = defineRelationsPart(schema, (r) => ({
  accounts: {
    users: r.one.users({
      from: r.accounts.userId,
      to: r.users.id,
    }),
  },
  users: {
    accounts: r.many.accounts({
      from: r.users.id,
      to: r.accounts.userId,
    }),
    projects: r.many.projects({
      from: r.users.id,
      to: r.projects.userId,
    }),
    sessions: r.many.sessions({
      from: r.users.id,
      to: r.sessions.userId,
    }),
    apiKeys: r.many.apiKeys({
      from: r.users.id,
      to: r.apiKeys.userId,
    }),
  },
  apiKeys: {
    user: r.one.users({
      from: r.apiKeys.userId,
      to: r.users.id,
      optional: false,
    }),
  },
  sessions: {
    users: r.one.users({
      from: r.sessions.userId,
      to: r.users.id,
    }),
  },
}))

const projectRelations = defineRelationsPart(schema, (r) => ({
  projects: {
    draftDataset: r.one.datasets({
      from: r.projects.id,
      to: r.datasets.projectId,
    }),
    trainingRuns: r.many.trainingRuns({
      from: r.projects.id,
      to: r.trainingRuns.projectId,
    }),
    user: r.one.users({
      from: r.projects.userId,
      to: r.users.id,
    }),
  },
}))

const datasetRelations = defineRelationsPart(schema, (r) => ({
  datasets: {
    project: r.one.projects({
      from: r.datasets.projectId,
      to: r.projects.id,
    }),
    draft: r.one.datasetVersions({
      from: r.datasets.projectId,
      to: r.datasetVersions.datasetId,
      where: {
        versionTag: { isNull: true },
      },
    }),
    // All immutable versions (snapshots) of this dataset
    versions: r.many.datasetVersions({
      from: r.datasets.projectId,
      to: r.datasetVersions.datasetId,
      where: {
        versionTag: { isNotNull: true },
      },
    }),
    classes: r.many.labelClasses({
      from: r.datasets.projectId,
      to: r.labelClasses.datasetId,
    }),
    // The project-wide deduplicated item pool
    items: r.many.datasetItems({
      from: r.datasets.projectId,
      to: r.datasetItems.datasetId,
    }),
  },

  labelClasses: {
    dataset: r.one.datasets({
      from: r.labelClasses.datasetId,
      to: r.datasets.projectId,
      optional: false,
    }),
  },

  datasetVersions: {
    dataset: r.one.datasets({
      from: r.datasetVersions.datasetId,
      to: r.datasets.projectId,
      optional: false,
    }),
    // Pool items this snapshot (or the draft) includes, with their split.
    items: r.many.datasetVersionItems({
      from: r.datasetVersions.id,
      to: r.datasetVersionItems.versionId,
    }),
    runs: r.many.trainingRuns({
      from: r.datasetVersions.id,
      to: r.trainingRuns.datasetVersionId,
    }),
  },

  datasetVersionItems: {
    version: r.one.datasetVersions({
      from: r.datasetVersionItems.versionId,
      to: r.datasetVersions.id,
      optional: false,
    }),
    item: r.one.datasetItems({
      from: r.datasetVersionItems.itemId,
      to: r.datasetItems.id,
      optional: false,
    }),
  },
}))

const datasetItemsRelations = defineRelationsPart(schema, (r) => ({
  datasetItems: {
    dataset: r.one.datasets({
      from: r.datasetItems.datasetId,
      to: r.datasets.projectId,
      optional: false,
    }),
    versionMemberships: r.many.datasetVersionItems({
      from: r.datasetItems.id,
      to: r.datasetVersionItems.itemId,
    }),

    // Note: textFeatures, visionFeatures, audioFeatures, tabularFeatures are on-to-one, so only one can exist per item.
    // This is handled by the database schema (uniqueness constraints).
    textFeatures: r.one.textFeatures({
      from: r.datasetItems.id,
      to: r.textFeatures.itemId,
    }),
    visionFeatures: r.one.visionFeatures({
      from: r.datasetItems.id,
      to: r.visionFeatures.itemId,
    }),
    audioFeatures: r.one.audioFeatures({
      from: r.datasetItems.id,
      to: r.audioFeatures.itemId,
    }),
    tabularFeatures: r.one.tabularFeatures({
      from: r.datasetItems.id,
      to: r.tabularFeatures.itemId,
    }),
    annotations: r.many.annotations({
      from: r.datasetItems.id,
      to: r.annotations.itemId,
    }),
  },

  textFeatures: {
    item: r.one.datasetItems({
      from: r.textFeatures.itemId,
      to: r.datasetItems.id,
    }),
  },

  visionFeatures: {
    item: r.one.datasetItems({
      from: r.visionFeatures.itemId,
      to: r.datasetItems.id,
    }),
  },

  audioFeatures: {
    item: r.one.datasetItems({
      from: r.audioFeatures.itemId,
      to: r.datasetItems.id,
    }),
  },

  tabularFeatures: {
    item: r.one.datasetItems({
      from: r.tabularFeatures.itemId,
      to: r.datasetItems.id,
    }),
  },

  annotations: {
    item: r.one.datasetItems({
      from: r.annotations.itemId,
      to: r.datasetItems.id,
      optional: false,
    }),
  },
}))

const trainingRelations = defineRelationsPart(schema, (r) => ({
  sweeps: {
    project: r.one.projects({
      from: r.sweeps.projectId,
      to: r.projects.id,
      optional: false,
    }),
    datasetVersion: r.one.datasetVersions({
      from: r.sweeps.datasetVersionId,
      to: r.datasetVersions.id,
      optional: false,
    }),
    trials: r.many.trainingRuns({
      from: r.sweeps.id,
      to: r.trainingRuns.sweepId,
    }),
  },
  trainingRuns: {
    project: r.one.projects({
      from: r.trainingRuns.projectId,
      to: r.projects.id,
    }),
    datasetVersion: r.one.datasetVersions({
      from: r.trainingRuns.datasetVersionId,
      to: r.datasetVersions.id,
    }),
    sweep: r.one.sweeps({
      from: r.trainingRuns.sweepId,
      to: r.sweeps.id,
    }),
    metrics: r.many.trainingMetrics({
      from: r.trainingRuns.id,
      to: r.trainingMetrics.trainingRunId,
    }),
    exports: r.many.modelExports({
      from: r.trainingRuns.id,
      to: r.modelExports.runId,
    }),
    inferenceJobs: r.many.inferenceJobs({
      from: r.trainingRuns.id,
      to: r.inferenceJobs.runId,
    }),
    evaluation: r.one.runEvaluations({
      from: r.trainingRuns.id,
      to: r.runEvaluations.runId,
    }),
  },
  trainingMetrics: {
    run: r.one.trainingRuns({
      from: r.trainingMetrics.trainingRunId,
      to: r.trainingRuns.id,
    }),
  },
  runEvaluations: {
    run: r.one.trainingRuns({
      from: r.runEvaluations.runId,
      to: r.trainingRuns.id,
      optional: false,
    }),
  },
  modelExports: {
    run: r.one.trainingRuns({
      from: r.modelExports.runId,
      to: r.trainingRuns.id,
      optional: false,
    }),
    user: r.one.users({
      from: r.modelExports.userId,
      to: r.users.id,
      optional: false,
    }),
  },
  inferenceJobs: {
    run: r.one.trainingRuns({
      from: r.inferenceJobs.runId,
      to: r.trainingRuns.id,
      optional: false,
    }),
  },
}))

export default {
  ...authRelations,
  ...projectRelations,
  ...datasetRelations,
  ...datasetItemsRelations,
  ...trainingRelations,
}

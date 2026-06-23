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
}))

const projectRelations = defineRelationsPart(schema, (r) => ({
  projects: {
    draftDataset: r.one.datasets(),
    trainingRuns: r.many.trainingRuns(),
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
    classes: r.many.labelClasses(),
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
    splits: r.many.datasetSplits(),
    runs: r.many.trainingRuns(),
  },

  datasetSplits: {
    version: r.one.datasetVersions({
      from: r.datasetSplits.datasetVersionId,
      to: r.datasetVersions.id,
      optional: false,
    }),
    items: r.many.datasetItems(),
  },
}))

const datasetItemsRelations = defineRelationsPart(schema, (r) => ({
  datasetItems: {
    split: r.one.datasetSplits({
      from: r.datasetItems.datasetSplitId,
      to: r.datasetSplits.id,
      optional: false,
    }),

    // Note: textFeatures, visionFeatures, audioFeatures, tabularFeatures are on-to-one, so only one can exist per item.
    // This is handled by the database schema (uniqueness constraints).
    textFeatures: r.one.textFeatures(),
    visionFeatures: r.one.visionFeatures(),
    audioFeatures: r.one.audioFeatures(),
    tabularFeatures: r.one.tabularFeatures(),
    annotations: r.many.annotations(),
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
  trainingRuns: {
    project: r.one.projects({
      from: r.trainingRuns.projectId,
      to: r.projects.id,
    }),
    datasetVersion: r.one.datasetVersions({
      from: r.trainingRuns.datasetVersionId,
      to: r.datasetVersions.id,
    }),
    metrics: r.many.trainingMetrics(),
  },
  trainingMetrics: {
    run: r.one.trainingRuns({
      from: r.trainingMetrics.trainingRunId,
      to: r.trainingRuns.id,
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

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import {
  deleteFile,
  deletePrefix,
  evaluationPrefix,
  snapshotManifestKey,
  snapshotParquetKey,
  trainingConfigKey,
  trainingLogsKey,
  trainingResultsPrefix,
} from '@server/lib/storage'

/** Deletes every S3 object belonging to a project: pool files, snapshots, training artifacts, and exports. */
export async function cleanupProjectStorage(projectId: string): Promise<void> {
  const [items, versions, runs] = await Promise.all([
    db.query.datasetItems.findMany({ where: { datasetId: projectId }, columns: { storageUrl: true } }),
    db.query.datasetVersions.findMany({ where: { datasetId: projectId }, columns: { id: true, versionTag: true } }),
    db.query.trainingRuns.findMany({ where: { projectId }, columns: { id: true } }),
  ])

  const tasks: Promise<unknown>[] = []

  for (const item of items) {
    if (item.storageUrl) tasks.push(deleteFile(CONSTANTS.BUCKET_DATASETS, item.storageUrl).catch(() => {}))
  }
  for (const version of versions) {
    tasks.push(...cleanupVersionStorageTasks(version.id, version.versionTag))
  }
  for (const run of runs) {
    tasks.push(...cleanupRunStorageTasks(run.id))
  }

  await Promise.allSettled(tasks)
}

function cleanupVersionStorageTasks(versionId: string, versionTag: string | null): Promise<unknown>[] {
  // The draft (versionTag === null) has no parquet/manifest — only real
  // snapshots do (see lib/snapshot.ts).
  if (versionTag === null) return []
  return [
    deleteFile(CONSTANTS.BUCKET_DATASETS, snapshotParquetKey(versionId)).catch(() => {}),
    deleteFile(CONSTANTS.BUCKET_DATASETS, snapshotManifestKey(versionId)).catch(() => {}),
  ]
}

/** Deletes a single snapshot version's S3 objects. Safe to call for the draft too — it's a no-op. */
export async function cleanupVersionStorage(versionId: string, versionTag: string | null): Promise<void> {
  await Promise.allSettled(cleanupVersionStorageTasks(versionId, versionTag))
}

function cleanupRunStorageTasks(runId: string): Promise<unknown>[] {
  return [
    deleteFile(CONSTANTS.BUCKET_TRAINING, trainingConfigKey(runId)).catch(() => {}),
    deletePrefix(CONSTANTS.BUCKET_TRAINING, trainingResultsPrefix(runId)).catch(() => {}),
    deleteFile(CONSTANTS.BUCKET_TRAINING, trainingLogsKey(runId)).catch(() => {}),
    deletePrefix(CONSTANTS.BUCKET_TRAINING, evaluationPrefix(runId)).catch(() => {}),
    // Covers model.{format}, expected.json, and bundles/*.zip in one sweep.
    deletePrefix(CONSTANTS.BUCKET_MODELS, `${runId}/`).catch(() => {}),
  ]
}

/** Deletes a single training run's S3 objects (config, results, logs, exported models/bundles). */
export async function cleanupRunStorage(runId: string): Promise<void> {
  await Promise.allSettled(cleanupRunStorageTasks(runId))
}

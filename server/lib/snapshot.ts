/**
 * Builds a version's dataset.parquet from its pool-item membership.
 *
 * The parquet holds S3 URIs and label/scalar values, not raw bytes — files
 * stay in the content-addressed pool and Ludwig reads them through s3fs, so
 * a snapshot is metadata-sized. That's why this runs inline in the gateway
 * rather than as a NATS task.
 */

import { GetObjectCommand } from '@aws-sdk/client-s3'
import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { datasetVersions } from '@server/db/schema'
import type { ProjectTask, SplitType } from '@server/lib/enums'
import { s3, snapshotManifestKey, snapshotParquetKey, uploadFile } from '@server/lib/storage'
import type { ColumnSpec, SnapshotContext } from '@server/lib/tasks'
import { getTaskDescriptor } from '@server/lib/tasks'
import { record } from '@server/lib/telemetry'
import { eq } from 'drizzle-orm'
import { ByteWriter, parquetWriteRows } from 'hyparquet-writer'

/**
 * Ludwig's `fixed` split preprocessing type (see lib/ludwig/compile.ts)
 * requires an integer column — its splitter does `column.astype(np.int8)`
 * then partitions on the literal values 0/1/2 (train/validation/test), so
 * the human-readable `split` string column (kind: 'split', used by
 * ai_service/tasks/export.py to pick a test-split golden sample) can't
 * double as that column. This one is synthetic: appended here, not
 * declared by any task in the registry, so it never appears in a task's
 * own input/output features.
 */
const SPLIT_INDEX_COLUMN = CONSTANTS.SPLIT_INDEX_COLUMN_NAME
const SPLIT_INDEX: Record<SplitType, number> = { train: 0, validation: 1, test: 2 }

/**
 * Synthetic passthrough column carrying each row's pool item id — Ludwig
 * never references it (not declared by any task's inputFeatures/
 * outputFeatures), but the worker's evaluation step reads it back off its
 * own predictions to join a misclassified row back to the item it came
 * from. Same passthrough pattern as SPLIT_INDEX_COLUMN above.
 */
const ITEM_ID_COLUMN = CONSTANTS.ITEM_ID_COLUMN_NAME

interface ManifestV1 {
  itemCount: number
  classCount: number
  classes: string[]
  /**
   * Item count per class name, classification tasks only — lets the
   * compiler build balanced class weights (see lib/ludwig/compile.ts's
   * `useClassWeights`) without a separate DB round trip at train-dispatch
   * time. Keyed by class name (not id) for the same reason class weights
   * themselves are: Ludwig resolves a name-keyed `class_weights` dict
   * against its own vocabulary internally, so nothing here ever needs to
   * know the index Ludwig assigns to each class.
   */
  classCounts?: Record<string, number>
  columns: ColumnSpec[]
  createdAt: string
}

/**
 * Build the parquet + manifest for a version and flip its status to
 * ready/failed. Call this after inserting the version's membership rows
 * (see the version-creation route in datasets.ts).
 */
export async function buildSnapshot(versionId: string): Promise<void> {
  return record('snapshot.build', async (span) => {
    span.setAttribute('theseus.version_id', versionId)
    try {
      const version = await db.query.datasetVersions.findFirst({
        where: { id: versionId },
        with: { dataset: { with: { project: { columns: { task: true } } } } },
      })
      if (!version) throw new Error(`Version ${versionId} not found`)
      if (!version.dataset.project) throw new Error(`Version ${versionId}'s project not found`)

      const task = getTaskDescriptor(version.dataset.project.task as ProjectTask)
      const members = await db.query.datasetVersionItems.findMany({
        where: { versionId },
        with: {
          item: {
            with: { textFeatures: true, tabularFeatures: true, annotations: true },
          },
        },
      })

      const classRows = await db.query.labelClasses.findMany({
        where: { datasetId: version.dataset.projectId, isActive: true },
      })
      const classNameById = new Map(classRows.map((c) => [c.classId, c.name]))

      // Tabular tasks don't declare their scalar columns statically — derive
      // them from the first item's dynamic feature keys.
      const scalarColumns =
        task.columns.some((c) => c.kind === 'scalar') || task.modality !== 'tabular'
          ? []
          : Object.keys(members[0]?.item.tabularFeatures?.featuresJson ?? {}).map((name) => ({
              name,
              kind: 'scalar' as const,
            }))
      const columns: ColumnSpec[] = [
        ...task.columns,
        ...scalarColumns,
        { name: SPLIT_INDEX_COLUMN, kind: 'split_index' },
        { name: ITEM_ID_COLUMN, kind: 'item_id' },
      ]

      const rows = members.map((m) => {
        const row: Record<string, unknown> = {}
        for (const col of columns) {
          row[col.name] = resolveColumnValue(col, m, classNameById)
        }
        return row
      })

      // Only classification tasks have a meaningful class distribution — a
      // regression 'target' column also has `kind: 'label'` but holds numbers.
      const labelColumn = columns.find((c) => c.kind === 'label')
      const classCounts: Record<string, number> | undefined =
        labelColumn && task.annotation.requiresLabelClasses
          ? rows.reduce<Record<string, number>>((counts, row) => {
              const value = row[labelColumn.name]
              if (typeof value === 'string') counts[value] = (counts[value] ?? 0) + 1
              return counts
            }, {})
          : undefined

      const writer = new ByteWriter()
      parquetWriteRows({
        writer,
        rows,
        columns: columns.map((c) => ({ name: c.name, type: columnParquetType(c) })),
      })
      const buffer = new Uint8Array(writer.getBuffer())

      const classCount = classRows.length
      const manifest: ManifestV1 = {
        itemCount: members.length,
        classCount,
        classes: classRows.map((c) => c.name),
        classCounts,
        columns,
        createdAt: new Date().toISOString(),
      }

      const parquetKey = snapshotParquetKey(versionId)
      await uploadFile(CONSTANTS.BUCKET_DATASETS, parquetKey, buffer, 'application/octet-stream')
      await uploadFile(
        CONSTANTS.BUCKET_DATASETS,
        snapshotManifestKey(versionId),
        new TextEncoder().encode(JSON.stringify(manifest, null, 2)),
        'application/json',
      )

      await db
        .update(datasetVersions)
        .set({
          status: 'ready',
          itemCount: members.length,
          classCount,
          parquetKey,
          builtAt: new Date(),
        })
        .where(eq(datasetVersions.id, versionId))
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      console.error(`[snapshot] Build failed for version ${versionId}:`, e)
      await db
        .update(datasetVersions)
        .set({ status: 'failed', failedMessage: message })
        .where(eq(datasetVersions.id, versionId))
    }
  })
}

export type MemberWithItem = Awaited<ReturnType<typeof db.query.datasetVersionItems.findMany>>[number] & {
  item: {
    storageUrl: string | null
    textFeatures: { rawText: string } | null
    tabularFeatures: { featuresJson: unknown } | null
    annotations: {
      classId: string | null
      labelStructured: unknown
      annotationType: string
      labelTextSequence: string | null
    }[]
  }
}

export function resolveColumnValue(col: ColumnSpec, member: MemberWithItem, classNameById: Map<string, string>): unknown {
  switch (col.kind) {
    case 'storage_uri':
      return member.item.storageUrl ? `s3://${CONSTANTS.BUCKET_DATASETS}/${member.item.storageUrl}` : null
    case 'inline_text':
      return member.item.textFeatures?.rawText ?? null
    case 'split':
      return member.splitType
    case 'split_index':
      return SPLIT_INDEX[member.splitType]
    case 'item_id':
      return member.itemId
    case 'label': {
      const classification = member.item.annotations.find((a) => a.classId !== null)
      if (classification?.classId) return classNameById.get(classification.classId) ?? null
      // Regression targets: stored as { value: number } in labelStructured
      // since annotations were designed classification-first.
      const structured = member.item.annotations[0]?.labelStructured as { value?: number } | undefined
      return structured?.value ?? null
    }
    case 'scalar': {
      const features = member.item.tabularFeatures?.featuresJson as Record<string, unknown> | undefined
      return features?.[col.name] ?? null
    }
    case 'text_sequence_label': {
      const sequence = member.item.annotations.find((a) => a.annotationType === 'text_sequence')
      return sequence?.labelTextSequence ?? null
    }
  }
}

function columnParquetType(col: ColumnSpec): 'STRING' | 'DOUBLE' | 'INT32' {
  if (col.kind === 'scalar') return 'DOUBLE'
  if (col.kind === 'split_index') return 'INT32'
  return 'STRING'
}

/** Read back a snapshot's manifest — used by training dispatch to rebuild the SnapshotContext for the Ludwig compiler. */
export async function readSnapshotManifest(versionId: string): Promise<SnapshotContext> {
  const response = await s3.send(
    new GetObjectCommand({ Bucket: CONSTANTS.BUCKET_DATASETS, Key: snapshotManifestKey(versionId) }),
  )
  const manifest = JSON.parse(await response.Body!.transformToString()) as ManifestV1
  return { columns: manifest.columns, labelClassNames: manifest.classes, classCounts: manifest.classCounts }
}

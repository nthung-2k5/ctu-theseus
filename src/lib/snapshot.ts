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
import type { ProjectTask } from '@server/lib/enums'
import { s3, snapshotManifestKey, snapshotParquetKey, uploadFile } from '@server/lib/storage'
import type { ColumnSpec, SnapshotContext } from '@server/lib/tasks'
import { getTaskDescriptor } from '@server/lib/tasks'
import { record } from '@server/lib/telemetry'
import { eq } from 'drizzle-orm'
import { ByteWriter, parquetWriteRows } from 'hyparquet-writer'

interface ManifestV1 {
  itemCount: number
  classCount: number
  classes: string[]
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
      const columns: ColumnSpec[] = [...task.columns, ...scalarColumns]

      const rows = members.map((m) => {
        const row: Record<string, unknown> = {}
        for (const col of columns) {
          row[col.name] = resolveColumnValue(col, m, classNameById)
        }
        return row
      })

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

type MemberWithItem = Awaited<ReturnType<typeof db.query.datasetVersionItems.findMany>>[number] & {
  item: {
    storageUrl: string | null
    textFeatures: { rawText: string } | null
    tabularFeatures: { featuresJson: unknown } | null
    annotations: { classId: string | null; labelStructured: unknown }[]
  }
}

function resolveColumnValue(col: ColumnSpec, member: MemberWithItem, classNameById: Map<string, string>): unknown {
  switch (col.kind) {
    case 'storage_uri':
      return member.item.storageUrl ? `s3://${CONSTANTS.BUCKET_DATASETS}/${member.item.storageUrl}` : null
    case 'inline_text':
      return member.item.textFeatures?.rawText ?? null
    case 'split':
      return member.splitType
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
  }
}

function columnParquetType(col: ColumnSpec): 'STRING' | 'DOUBLE' {
  return col.kind === 'scalar' ? 'DOUBLE' : 'STRING'
}

/** Read back a snapshot's manifest — used by training dispatch to rebuild the SnapshotContext for the Ludwig compiler. */
export async function readSnapshotManifest(versionId: string): Promise<SnapshotContext> {
  const response = await s3.send(
    new GetObjectCommand({ Bucket: CONSTANTS.BUCKET_DATASETS, Key: snapshotManifestKey(versionId) }),
  )
  const manifest = JSON.parse(await response.Body!.transformToString()) as ManifestV1
  return { columns: manifest.columns, labelClassNames: manifest.classes }
}

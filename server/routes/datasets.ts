/**
 * Dataset routes – manages the pool/version/item hierarchy under a project's
 * single 1:1 dataset.
 *
 * The dataset itself is created atomically with the project (see projects.ts).
 * Items live in a project-wide deduplicated pool (dataset_items); a version
 * (draft or snapshot) is just a set of pool items with a split assignment
 * (dataset_version_items). These routes manage everything below the dataset
 * level:
 *
 *   POST   /api/projects/:projectId/versions          – Snapshot the draft into an immutable version
 *   GET    /api/versions/:versionId                   – Get version with per-split item counts
 *   DELETE /api/versions/:versionId                   – Delete a version
 *
 *   GET    /api/projects/:projectId/items              – Paginated pool items (draft or a given version)
 *   POST   /api/projects/:projectId/items               – Add items to the pool + draft, with a split
 *   POST   /api/projects/:projectId/upload               – Upload files to the pool + draft, with a split
 *   DELETE /api/projects/:projectId/items               – Bulk-delete pool items
 *   PATCH  /api/projects/:projectId/items/split          – Bulk-reassign the draft split of pool items
 *   POST   /api/projects/:projectId/items/classify        – Bulk-assign a label class to pool items
 *   POST   /api/projects/:projectId/items/auto-split       – Randomly reassign every draft item's split by ratio (stratified by class for classification tasks)
 *   DELETE /api/items/:itemId                          – Delete a single item from the pool
 *
 *   POST   /api/items/:itemId/annotations             – Add annotation to an item
 *   GET    /api/items/:itemId/annotations             – List annotations for an item
 *   PATCH  /api/annotations/:annotationId             – Update an annotation (re-labeling)
 *   DELETE /api/annotations/:annotationId             – Delete an annotation
 *
 * Snapshotting copies pool-item membership into the new version, then kicks
 * off an async parquet build (src/lib/snapshot.ts) — poll GET
 * /versions/:versionId for status: 'ready' | 'failed'.
 */

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import {
  annotations,
  audioFeatures,
  datasetItems,
  datasetVersionItems,
  datasetVersions,
  labelClasses,
  tabularFeatures,
  textFeatures,
  visionFeatures,
} from '@server/db/schema'
import { cleanupVersionStorage } from '@server/lib/cleanup'
import { type SplitType, SplitTypes } from '@server/lib/enums'
import { readImageDimensions } from '@server/lib/image-size'
import { buildSnapshot } from '@server/lib/snapshot'
import { deleteFile, getDownloadUrl, uploadToPool } from '@server/lib/storage'
import { getTaskDescriptor, isClassificationTask } from '@server/lib/tasks'
import {
  and,
  asc,
  avg,
  count,
  countDistinct,
  desc,
  eq,
  inArray,
  isNotNull,
  isNull,
  max,
  min,
  notInArray,
  sql,
} from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

const SortOptions = ['newest', 'oldest', 'filename'] as const
type SortOption = (typeof SortOptions)[number]

// A class with fewer than this many items can't appear in all three splits
// (auto-split) and can't be meaningfully evaluated on its own (health check)
// — shared threshold so the two don't drift apart.
const MIN_ITEMS_PER_CLASS = 3

export const datasetRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)

  /* ================================================================== */
  /*  VERSION-LEVEL ENDPOINTS                                           */
  /* ================================================================== */

  /* ── Snapshot the draft's current pool membership into a new immutable version ── */
  .post(
    '/projects/:projectId/versions',
    async ({ project, draft, body }) => {
      const [version] = await db
        .insert(datasetVersions)
        .values({
          datasetId: project.id,
          versionTag: body.versionTag,
          status: 'building',
        })
        .returning()

      const draftMembers = await db.query.datasetVersionItems.findMany({
        where: { versionId: draft.id },
      })

      if (draftMembers.length > 0) {
        await db.insert(datasetVersionItems).values(
          draftMembers.map((m) => ({
            versionId: version.id,
            itemId: m.itemId,
            splitType: m.splitType,
          })),
        )
      }

      const [updated] = await db
        .update(datasetVersions)
        .set({ itemCount: draftMembers.length })
        .where(eq(datasetVersions.id, version.id))
        .returning()

      // Fire-and-forget: the parquet is metadata-sized (S3 URIs, not raw
      // bytes), so building it inline is cheap. The route returns
      // immediately with status: 'building'; poll GET /versions/:id.
      buildSnapshot(version.id).catch((e) => console.error(`[snapshot] Unhandled error building ${version.id}:`, e))

      return status(202, { version: updated })
    },
    {
      draftBelongToUser: true,
      body: t.Object({
        versionTag: t.String({ minLength: 1, maxLength: 50 }),
      }),
    },
  )

  /* ── Get a version with its per-split item counts ── */
  .get(
    '/versions/:versionId',
    async ({ version }) => {
      const members = await db.query.datasetVersionItems.findMany({
        where: { versionId: version.id },
        columns: { splitType: true },
      })
      const splits = SplitTypes.map((splitType) => ({
        splitType,
        itemCount: members.filter((m) => m.splitType === splitType).length,
      }))

      return {
        version: {
          ...version,
          dataset: { projectId: version.dataset.projectId, modality: version.dataset.modality },
          splits,
        },
      }
    },
    { versionBelongToUser: true },
  )

  /* ── Delete a version (cannot delete draft) ── */
  .delete(
    '/versions/:versionId',
    async ({ version }) => {
      if (version.versionTag === null) return status(400, 'Cannot delete draft version')

      await cleanupVersionStorage(version.id, version.versionTag)
      await db.delete(datasetVersions).where(eq(datasetVersions.id, version.id))
      return status(204)
    },
    { versionBelongToUser: true },
  )

  /* ================================================================== */
  /*  POOL & ITEM-LEVEL ENDPOINTS                                       */
  /* ================================================================== */

  /* ── List items in a project's pool (draft by default, or a given version) ── */
  .get(
    '/projects/:projectId/items',
    async ({ params, query, project }) => {
      const version = await db.query.datasetVersions.findFirst({
        where: query.versionId
          ? { id: query.versionId, datasetId: params.projectId }
          : { datasetId: params.projectId, versionTag: { isNull: true } },
      })
      if (!version) return status(404, 'Version not found')

      // "Labeled" means "carries this task's own ground-truth annotation
      // type" — classification for most tasks, but text_sequence for
      // captioning/ASR. Hardcoding 'classification' here previously left
      // labeledCount/unassignedCount permanently wrong (always 0/all) for
      // any text_sequence-annotated task.
      const groundTruthType = getTaskDescriptor(project.task).annotation.type

      const page = Math.max(1, query.page ?? 1)
      const perPage = Math.min(1000, Math.max(1, query.perPage ?? 30))
      const offset = (page - 1) * perPage

      // t.Optional(t.UnionEnum(...)) silently defaults to the first enum
      // value when the query param is absent instead of staying undefined,
      // so `split` and `sort` are validated as plain strings and checked
      // manually.
      const split: SplitType | undefined = (SplitTypes as readonly string[]).includes(query.split ?? '')
        ? (query.split as SplitType)
        : undefined

      const sort: SortOption = (SortOptions as readonly string[]).includes(query.sort ?? '')
        ? (query.sort as SortOption)
        : 'newest'
      const orderByClause =
        sort === 'oldest'
          ? asc(datasetItems.createdAt)
          : sort === 'filename'
            ? asc(datasetItems.externalId)
            : desc(datasetItems.createdAt)

      // classId isn't a column on dataset_version_items — resolve it to the
      // set of item ids carrying that class's classification annotation
      // first, then filter membership by itemId IN (...). The sentinel
      // 'unassigned' inverts this: resolve every item that HAS a
      // classification annotation (of any class) and filter it OUT instead.
      // Both branches join through dataset_items so the scan (and the
      // resulting id list) is scoped to this project — annotations has no
      // datasetId column of its own, and an unscoped findMany here previously
      // read every classification annotation in the entire database on every
      // request that hit this filter.
      let classItemIds: string[] | undefined
      let classItemIdsMode: 'in' | 'notIn' = 'in'
      if (query.classId === 'unassigned') {
        const rows = await db
          .select({ itemId: annotations.itemId })
          .from(annotations)
          .innerJoin(datasetItems, eq(annotations.itemId, datasetItems.id))
          .where(and(eq(datasetItems.datasetId, params.projectId), eq(annotations.annotationType, 'classification')))
        classItemIds = rows.map((r) => r.itemId)
        classItemIdsMode = 'notIn'
      } else if (query.classId) {
        const rows = await db
          .select({ itemId: annotations.itemId })
          .from(annotations)
          .innerJoin(datasetItems, eq(annotations.itemId, datasetItems.id))
          .where(
            and(
              eq(datasetItems.datasetId, params.projectId),
              eq(annotations.classId, query.classId),
              eq(annotations.annotationType, 'classification'),
            ),
          )
        classItemIds = rows.map((r) => r.itemId)
      }

      // Same resolve-to-itemIds approach as classId: externalId lives on
      // dataset_items, not dataset_version_items, so it can't be filtered
      // directly in the count/labeled/members queries below without joining
      // dataset_items into all three.
      let searchItemIds: string[] | undefined
      if (query.search) {
        const rows = await db.query.datasetItems.findMany({
          where: { datasetId: params.projectId, externalId: { ilike: `%${query.search}%` } },
          columns: { id: true },
        })
        searchItemIds = rows.map((r) => r.id)
      }

      // Split/search-scoped, but deliberately WITHOUT the classId filter —
      // this is what the class-count breakdown below groups over, so every
      // class's count reflects the current split/search regardless of which
      // class (if any) is currently selected in the filter.
      const baseClauses = [
        eq(datasetVersionItems.versionId, version.id),
        ...(split ? [eq(datasetVersionItems.splitType, split)] : []),
        ...(searchItemIds ? [inArray(datasetVersionItems.itemId, searchItemIds)] : []),
      ]

      const rawClauses = [
        ...baseClauses,
        ...(classItemIds
          ? [
              classItemIdsMode === 'notIn'
                ? notInArray(datasetVersionItems.itemId, classItemIds)
                : inArray(datasetVersionItems.itemId, classItemIds),
            ]
          : []),
      ]
      const [total, labeledRows, members, baseTotal, classCountRows, classifiedRows] = await Promise.all([
        db.$count(datasetVersionItems, and(...rawClauses)),
        // Distinct item count with the task's own ground-truth annotation
        // type, scoped to this version/split — powers the labeling-progress
        // indicator. Most stable tasks' ground truth is one
        // classification-type annotation per item (a class pick or a
        // regression target), but captioning/ASR tasks use text_sequence —
        // see `groundTruthType` above.
        db
          .select({ count: countDistinct(datasetVersionItems.itemId) })
          .from(datasetVersionItems)
          .innerJoin(
            annotations,
            and(eq(annotations.itemId, datasetVersionItems.itemId), eq(annotations.annotationType, groundTruthType)),
          )
          .where(and(...rawClauses)),
        // Sorting is on `dataset_items` columns (createdAt, externalId), which
        // aren't reachable from the relational query API's orderBy (it only
        // sees the root table, dataset_version_items) — so page/order the
        // membership rows with a plain join first, then hydrate the full item
        // rows (with their feature/annotation relations) in id order below.
        db
          .select({ itemId: datasetVersionItems.itemId, splitType: datasetVersionItems.splitType })
          .from(datasetVersionItems)
          .innerJoin(datasetItems, eq(datasetVersionItems.itemId, datasetItems.id))
          .where(and(...rawClauses))
          .orderBy(orderByClause)
          .limit(perPage)
          .offset(offset),
        // Per-class item counts for the class filter dropdown ("Cat (10)") —
        // scoped to split/search but not classId (see baseClauses above).
        db.$count(datasetVersionItems, and(...baseClauses)),
        db
          .select({ classId: annotations.classId, count: countDistinct(datasetVersionItems.itemId) })
          .from(datasetVersionItems)
          .innerJoin(
            annotations,
            and(eq(annotations.itemId, datasetVersionItems.itemId), eq(annotations.annotationType, 'classification')),
          )
          .where(and(...baseClauses))
          .groupBy(annotations.classId),
        db
          .select({ count: countDistinct(datasetVersionItems.itemId) })
          .from(datasetVersionItems)
          .innerJoin(
            annotations,
            and(eq(annotations.itemId, datasetVersionItems.itemId), eq(annotations.annotationType, groundTruthType)),
          )
          .where(and(...baseClauses)),
      ])

      const itemRows =
        members.length > 0
          ? await db.query.datasetItems.findMany({
              where: { id: { in: members.map((m) => m.itemId) } },
              with: {
                textFeatures: true,
                visionFeatures: true,
                audioFeatures: true,
                tabularFeatures: true,
                annotations: true,
              },
            })
          : []
      const itemById = new Map(itemRows.map((item) => [item.id, item]))

      const itemsWithUrls = await Promise.all(
        members.map(async (m) => {
          const item = itemById.get(m.itemId)
          if (!item) return null
          return {
            ...item,
            splitType: m.splitType,
            downloadUrl: item.storageUrl ? await getDownloadUrl(CONSTANTS.BUCKET_DATASETS, item.storageUrl, 3600) : null,
          }
        }),
      ).then((rows) => rows.filter((row) => row != null))

      const classCounts = classCountRows
        .filter((r): r is { classId: string; count: number } => r.classId != null)
        .map((r) => ({ classId: r.classId, count: r.count }))
      const unassignedCount = baseTotal - (classifiedRows[0]?.count ?? 0)

      return {
        items: itemsWithUrls,
        total,
        labeledCount: labeledRows[0]?.count ?? 0,
        classCounts,
        unassignedCount,
        page,
        perPage,
      }
    },
    {
      projectBelongToUser: true,
      query: t.Object({
        versionId: t.Optional(t.String({ format: 'uuid' })),
        split: t.Optional(t.String()),
        // Plain string, not a uuid format: also carries the 'unassigned'
        // sentinel (items with no classification annotation at all).
        classId: t.Optional(t.String()),
        search: t.Optional(t.String()),
        page: t.Optional(t.Numeric({ minimum: 1 })),
        perPage: t.Optional(t.Numeric({ minimum: 1, maximum: 1000 })),
        sort: t.Optional(t.String()),
      }),
    },
  )

  /* ── Add items to the pool + draft, with a split assignment ── */
  .post(
    '/projects/:projectId/items',
    async ({ project, draft, body }) => {
      const bodyClassIds = [
        ...new Set(body.items.flatMap((i) => i.annotations?.map((a) => a.classId).filter((c) => c != null) ?? [])),
      ]
      for (const classId of bodyClassIds) {
        if (!(await classInDataset(classId, project.id))) return status(400, `Unknown label class: ${classId}`)
      }

      const results = await Promise.allSettled(
        // Each item's item+features+annotations+membership rows are one
        // unit — a partial failure inside them (e.g. the item row commits
        // but its features row doesn't) would otherwise leave an orphaned
        // item with no features. The outer allSettled still lets one bad
        // item fail without failing the whole batch.
        body.items.map((itemData) =>
          db.transaction(async (tx) => {
            // Same content-addressed dedup as uploadToPool (server/lib/storage.ts)
            // for the file-upload route below — without it, re-importing the
            // same CSV (or two rows with identical text) creates two distinct
            // items, and auto-split can then scatter duplicates across train
            // and test, inflating test metrics on content the model already
            // trained on.
            const contentHash = hashItemContent(itemData)
            const existingItem = contentHash
              ? await tx.query.datasetItems.findFirst({ where: { datasetId: project.id, contentHash } })
              : null

            const item =
              existingItem ??
              (
                await tx
                  .insert(datasetItems)
                  .values({
                    datasetId: project.id,
                    externalId: itemData.externalId,
                    contentHash,
                  })
                  .returning()
              )[0]

            // Re-adding a soft-deleted item's exact content restores it (mirrors POST /upload).
            if (existingItem?.deletedAt) {
              await tx.update(datasetItems).set({ deletedAt: null }).where(eq(datasetItems.id, item.id))
            }

            // Only attach features/annotations to a freshly-created item — a
            // dedup hit means this content is already in the pool, possibly
            // already labeled differently, and feature tables are 1:1 on
            // itemId so re-inserting for an existing item would violate that.
            if (!existingItem) {
              if (itemData.textFeatures) {
                await tx.insert(textFeatures).values({
                  itemId: item.id,
                  rawText: itemData.textFeatures.rawText,
                  tokenCount: itemData.textFeatures.tokenCount,
                  languageCode: itemData.textFeatures.languageCode,
                  metaJson: itemData.textFeatures.metaJson,
                })
              }

              if (itemData.visionFeatures) {
                await tx.insert(visionFeatures).values({
                  itemId: item.id,
                  width: itemData.visionFeatures.width,
                  height: itemData.visionFeatures.height,
                  channels: itemData.visionFeatures.channels,
                  imageFormat: itemData.visionFeatures.imageFormat,
                  exifData: itemData.visionFeatures.exifData,
                })
              }

              if (itemData.audioFeatures) {
                await tx.insert(audioFeatures).values({
                  itemId: item.id,
                  durationSeconds: String(itemData.audioFeatures.durationSeconds),
                  sampleRateHz: itemData.audioFeatures.sampleRateHz,
                  channels: itemData.audioFeatures.channels,
                  audioCodec: itemData.audioFeatures.audioCodec,
                })
              }

              if (itemData.tabularFeatures) {
                await tx.insert(tabularFeatures).values({
                  itemId: item.id,
                  featuresJson: itemData.tabularFeatures.featuresJson,
                })
              }

              if (itemData.annotations?.length) {
                await tx.insert(annotations).values(
                  itemData.annotations.map((ann) => ({
                    itemId: item.id,
                    annotatorId: ann.annotatorId,
                    annotationType: ann.annotationType,
                    classId: ann.classId,
                    labelTextSequence: ann.labelTextSequence,
                    labelStructured: ann.labelStructured,
                    confidenceScore: ann.confidenceScore != null ? String(ann.confidenceScore) : undefined,
                  })),
                )
              }
            }

            await tx
              .insert(datasetVersionItems)
              .values({
                versionId: draft.id,
                itemId: item.id,
                splitType: itemData.split,
              })
              .onConflictDoNothing()

            return item
          }),
        ),
      )

      // Shaped instead of the raw PromiseSettledResult[] — callers (the
      // tabular CSV importer in particular) need a reliable created/failed
      // count without leaking Error objects to the client. `message` is
      // sanitized too: every failure here is a raw driver error (nothing in
      // this transaction throws its own friendly message), and the driver's
      // text can embed constraint/table/column names — not secret, but not
      // something to hand back verbatim to an API caller either. The actual
      // error is logged server-side.
      const created: (typeof datasetItems.$inferSelect)[] = []
      const failed: { index: number; message: string }[] = []
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          created.push(result.value)
        } else {
          console.error(`[items] Failed to create item at index ${index}:`, result.reason)
          failed.push({ index, message: toSafeItemErrorMessage(result.reason) })
        }
      })

      return { created, failed }
    },
    {
      draftBelongToUser: true,
      body: t.Object({
        items: t.Array(
          t.Object({
            split: t.UnionEnum(SplitTypes),
            // No `storageUrl` here deliberately — this route only ever creates
            // items from inline data (text/tabular). Accepting a client-chosen
            // S3 key would let one tenant point an item at another tenant's
            // pool/snapshot object; the only path that legitimately writes
            // storageUrl is uploadToPool (see POST /upload below), which
            // derives the key server-side from the uploaded content's hash.
            externalId: t.Optional(t.String()),
            textFeatures: t.Optional(
              t.Object({
                rawText: t.String(),
                tokenCount: t.Optional(t.Number()),
                languageCode: t.Optional(t.String()),
                metaJson: t.Optional(t.Any()),
              }),
            ),
            visionFeatures: t.Optional(
              t.Object({
                width: t.Number(),
                height: t.Number(),
                channels: t.Optional(t.Number()),
                imageFormat: t.Optional(t.UnionEnum(['jpeg', 'png'])),
                exifData: t.Optional(t.Any()),
              }),
            ),
            audioFeatures: t.Optional(
              t.Object({
                durationSeconds: t.Number(),
                sampleRateHz: t.Number(),
                channels: t.Optional(t.Number()),
                audioCodec: t.Optional(t.UnionEnum(['wav', 'mp3', 'flac', 'ogg'])),
              }),
            ),
            tabularFeatures: t.Optional(
              t.Object({
                featuresJson: t.Any(),
              }),
            ),
            annotations: t.Optional(
              t.Array(
                t.Object({
                  annotatorId: t.Optional(t.String()),
                  annotationType: t.UnionEnum([
                    'classification',
                    'bounding_box',
                    'segmentation_mask',
                    'text_sequence',
                    'token_tags',
                    'preference_rank',
                  ]),
                  classId: t.Optional(t.String({ format: 'uuid' })),
                  labelTextSequence: t.Optional(t.String()),
                  labelStructured: t.Optional(t.Any()),
                  confidenceScore: t.Optional(t.Number({ minimum: 0, maximum: 1 })),
                }),
              ),
            ),
          }),
          { minItems: 1 },
        ),
      }),
    },
  )

  /* ── Upload files to the pool + draft, with a split assignment (vision/audio) ── */
  .post(
    '/projects/:projectId/upload',
    async ({ project, draft, body }) => {
      if (body.classId && !(await classInDataset(body.classId, project.id)))
        return status(400, 'Unknown label class')

      const dataset = await db.query.datasets.findFirst({
        where: { projectId: project.id },
        columns: { modality: true },
      })
      if (!dataset) return status(404, 'Dataset not found')

      const results = await Promise.allSettled(
        body.files.map(async (file) => {
          const fileBytes = await file.bytes()
          const ext = file.name.includes('.') ? `.${file.name.split('.').pop()}` : ''
          // The S3 upload happens outside the transaction below — it's slow
          // network I/O that shouldn't hold a DB transaction open, and a
          // leftover pool object on a later DB failure is a harmless no-op
          // (content-addressed, so a retry just reuses it).
          const { key, hash, byteSize, isDuplicate } = await uploadToPool(project.id, fileBytes, ext, file.type)

          return db.transaction(async (tx) => {
            // The pool is content-addressed and deduplicated: if this exact
            // content already has an item, reuse it instead of violating the
            // (datasetId, contentHash) uniqueness constraint.
            const existingItem = await tx.query.datasetItems.findFirst({
              where: { datasetId: project.id, contentHash: hash },
            })

            const item =
              existingItem ??
              (
                await tx
                  .insert(datasetItems)
                  .values({
                    datasetId: project.id,
                    externalId: file.name,
                    storageUrl: key,
                    contentHash: hash,
                    byteSize,
                  })
                  .returning()
              )[0]

            // Re-uploading a soft-deleted item's exact content restores it.
            if (existingItem?.deletedAt) {
              await tx.update(datasetItems).set({ deletedAt: null }).where(eq(datasetItems.id, item.id))
            }

            if (!existingItem && dataset.modality === 'vision' && file.type.startsWith('image/')) {
              const dimensions = readImageDimensions(fileBytes)
              if (dimensions) {
                await tx.insert(visionFeatures).values({
                  itemId: item.id,
                  width: dimensions.width,
                  height: dimensions.height,
                  channels: 3,
                  imageFormat: file.type === 'image/png' ? 'png' : 'jpeg',
                })
              }
            }

            // Only label freshly-created items — a dedup hit means this
            // content is already in the pool, possibly already labeled
            // differently, and a same-batch upload shouldn't silently
            // overwrite that.
            if (!existingItem && body.classId) {
              await tx.insert(annotations).values({
                itemId: item.id,
                annotationType: 'classification',
                classId: body.classId,
              })
            }

            await tx
              .insert(datasetVersionItems)
              .values({
                versionId: draft.id,
                itemId: item.id,
                splitType: body.split,
              })
              .onConflictDoNothing()

            return { ...item, isDuplicate }
          })
        }),
      )

      return { results }
    },
    {
      draftBelongToUser: true,
      body: t.Object({
        split: t.UnionEnum(SplitTypes),
        files: t.Files({ maxSize: '50m' }),
        classId: t.Optional(t.String({ format: 'uuid' })),
      }),
    },
  )

  /* ── Delete a single item from the pool ── */
  .delete(
    '/items/:itemId',
    async ({ item }) => {
      await deleteItemFromPool(item.id, item.dataset.projectId)
      return status(204)
    },
    { itemBelongToUser: true },
  )

  /* ── Bulk-delete pool items ── */
  .delete(
    '/projects/:projectId/items',
    async ({ project, body }) => {
      const outcomes = await Promise.all(
        body.itemIds.map(async (itemId) => ({ itemId, outcome: await deleteItemFromPool(itemId, project.id) })),
      )
      return { results: outcomes }
    },
    {
      projectBelongToUser: true,
      body: t.Object({ itemIds: t.Array(t.String({ format: 'uuid' }), { minItems: 1, maxItems: 1000 }) }),
    },
  )

  /* ── Bulk-reassign the draft split of pool items ── */
  .patch(
    '/projects/:projectId/items/split',
    async ({ draft, body }) => {
      const updated = await db
        .update(datasetVersionItems)
        .set({ splitType: body.split })
        .where(and(eq(datasetVersionItems.versionId, draft.id), inArray(datasetVersionItems.itemId, body.itemIds)))
        .returning({ itemId: datasetVersionItems.itemId })

      return { updated: updated.map((u) => u.itemId) }
    },
    {
      draftBelongToUser: true,
      body: t.Object({ itemIds: t.Array(t.String({ format: 'uuid' }), { minItems: 1, maxItems: 1000 }), split: t.UnionEnum(SplitTypes) }),
    },
  )

  /* ── Bulk-assign a label class to pool items (creates or updates each item's classification annotation) ── */
  .post(
    '/projects/:projectId/items/classify',
    async ({ project, body }) => {
      if (!(await classInDataset(body.classId, project.id))) return status(400, 'Unknown label class')

      const results = await Promise.allSettled(
        body.itemIds.map((itemId) =>
          db.transaction(async (tx) => {
            const item = await tx.query.datasetItems.findFirst({ where: { id: itemId, datasetId: project.id } })
            if (!item) throw new Error('Item not found')

            // Single statement rather than read-then-write: with the partial
            // unique index on (item_id) where annotation_type='classification',
            // two concurrent classify calls can no longer both see "no existing
            // annotation" and both insert.
            await tx
              .insert(annotations)
              .values({ itemId: item.id, annotationType: 'classification', classId: body.classId })
              .onConflictDoUpdate({
                target: annotations.itemId,
                targetWhere: eq(annotations.annotationType, 'classification'),
                set: { classId: body.classId },
              })
          }),
        ),
      )

      const failed = results.filter((r) => r.status === 'rejected').length
      return { updated: results.length - failed, failed }
    },
    {
      projectBelongToUser: true,
      body: t.Object({ itemIds: t.Array(t.String({ format: 'uuid' }), { minItems: 1, maxItems: 1000 }), classId: t.String({ format: 'uuid' }) }),
    },
  )

  /* ── Randomly reassign every draft item's split, matching a train/validation/test ratio ── */
  .post(
    '/projects/:projectId/items/auto-split',
    async ({ draft, project, body }) => {
      const members = await db.query.datasetVersionItems.findMany({
        where: { versionId: draft.id },
        with: { item: { with: { annotations: true } } },
      })
      if (members.length === 0) return { updated: 0 }

      const ratios = body.ratios ?? { train: 80, validation: 10, test: 10 }
      const ratioTotal = ratios.train + ratios.validation + ratios.test
      if (ratioTotal <= 0) return status(400, 'Ratios must sum to a positive number')

      // Stratify by label class when the task actually has classes — grouping
      // an unlabeled/regression dataset by a nonexistent class would just be
      // one big "unassigned" group, i.e. the old flat behavior, so the flag
      // only changes anything for classification tasks.
      const stratify = body.stratify ?? isClassificationTask(project.task)

      // Un-stratified: one group holding every item, same as the old behavior.
      // Stratified: one group per label class (+ 'unassigned' for items with
      // no classification annotation yet), so a class present in the draft
      // can't be shuffled entirely out of validation/test.
      const groupsByKey = new Map<string, string[]>()
      for (const m of members) {
        const key = stratify ? (m.item.annotations.find((a) => a.classId !== null)?.classId ?? 'unassigned') : 'all'
        const arr = groupsByKey.get(key)
        if (arr) arr.push(m.itemId)
        else groupsByKey.set(key, [m.itemId])
      }

      const splitGroups: Record<SplitType, string[]> = { train: [], validation: [], test: [] }
      const warnings: string[] = []

      for (const [key, itemIds] of groupsByKey) {
        // Fisher-Yates — every item in this group lands in exactly one split, in random order.
        for (let i = itemIds.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1))
          ;[itemIds[i], itemIds[j]] = [itemIds[j], itemIds[i]]
        }

        const trainCount = Math.round((ratios.train / ratioTotal) * itemIds.length)
        const validationCount = Math.round((ratios.validation / ratioTotal) * itemIds.length)
        // test absorbs the rounding remainder so every item is assigned exactly once.
        splitGroups.train.push(...itemIds.slice(0, trainCount))
        splitGroups.validation.push(...itemIds.slice(trainCount, trainCount + validationCount))
        splitGroups.test.push(...itemIds.slice(trainCount + validationCount))

        if (stratify && key !== 'unassigned' && itemIds.length < MIN_ITEMS_PER_CLASS) {
          warnings.push(`Class ${key} has only ${itemIds.length} item(s) — it can't appear in all three splits.`)
        }
      }

      await db.transaction(async (tx) => {
        for (const splitType of SplitTypes) {
          if (splitGroups[splitType].length === 0) continue
          await tx
            .update(datasetVersionItems)
            .set({ splitType })
            .where(
              and(
                eq(datasetVersionItems.versionId, draft.id),
                inArray(datasetVersionItems.itemId, splitGroups[splitType]),
              ),
            )
        }
      })

      return {
        updated: members.length,
        stratified: stratify,
        splits: {
          train: splitGroups.train.length,
          validation: splitGroups.validation.length,
          test: splitGroups.test.length,
        },
        warnings,
      }
    },
    {
      draftBelongToUser: true,
      body: t.Object({
        stratify: t.Optional(t.Boolean()),
        ratios: t.Optional(
          t.Object({
            train: t.Number({ minimum: 0 }),
            validation: t.Number({ minimum: 0 }),
            test: t.Number({ minimum: 0 }),
          }),
        ),
      }),
    },
  )

  /* ── Dataset health / EDA report for the draft (what training would actually see) ── */
  .get(
    '/projects/:projectId/dataset/health',
    async ({ draft, project }) => {
      const modality = draft.dataset.modality
      // See the same-named const in GET /items above — most tasks' ground
      // truth is a classification annotation, but captioning/ASR tasks use
      // text_sequence.
      const groundTruthType = getTaskDescriptor(project.task).annotation.type

      const [{ itemCount }] = await db
        .select({ itemCount: count() })
        .from(datasetVersionItems)
        .where(eq(datasetVersionItems.versionId, draft.id))

      if (itemCount === 0) {
        return {
          health: {
            itemCount: 0,
            labeledCount: 0,
            unlabeledCount: 0,
            modality,
            classDistribution: [],
            smallClasses: [],
            duplicateContentHashes: 0,
            missingContentHash: 0,
            vision: null,
            audio: null,
            text: null,
            tabular: null,
          },
        }
      }

      const [{ labeledCount }] = await db
        .select({ labeledCount: countDistinct(annotations.itemId) })
        .from(annotations)
        .innerJoin(datasetVersionItems, eq(annotations.itemId, datasetVersionItems.itemId))
        .where(and(eq(datasetVersionItems.versionId, draft.id), eq(annotations.annotationType, groundTruthType)))

      let classDistribution: { classId: string; name: string; count: number }[] = []
      if (isClassificationTask(project.task)) {
        const rows = await db
          .select({ classId: labelClasses.classId, name: labelClasses.name, count: countDistinct(annotations.itemId) })
          .from(annotations)
          .innerJoin(datasetVersionItems, eq(annotations.itemId, datasetVersionItems.itemId))
          .innerJoin(labelClasses, eq(annotations.classId, labelClasses.classId))
          .where(and(eq(datasetVersionItems.versionId, draft.id), eq(annotations.annotationType, 'classification')))
          .groupBy(labelClasses.classId, labelClasses.name)
        classDistribution = rows.sort((a, b) => b.count - a.count)
      }
      const smallClasses = classDistribution.filter((c) => c.count < MIN_ITEMS_PER_CLASS)

      // Content-hash integrity is a pool-wide (not draft-scoped) invariant —
      // uploadToPool/hashItemContent's (datasetId, contentHash) dedup means
      // this should always read 0/0, so this is a live assurance check
      // rather than an actionable filter.
      const dupRows = await db
        .select({ contentHash: datasetItems.contentHash, dupCount: count() })
        .from(datasetItems)
        .where(
          and(eq(datasetItems.datasetId, project.id), isNull(datasetItems.deletedAt), isNotNull(datasetItems.contentHash)),
        )
        .groupBy(datasetItems.contentHash)
        .having(sql`count(*) > 1`)
      const [{ missingContentHash }] = await db
        .select({ missingContentHash: count() })
        .from(datasetItems)
        .where(
          and(eq(datasetItems.datasetId, project.id), isNull(datasetItems.deletedAt), isNull(datasetItems.contentHash)),
        )

      let vision: {
        count: number
        width: { min: number; max: number; avg: number }
        height: { min: number; max: number; avg: number }
        formats: Record<string, number>
      } | null = null
      let audio: {
        count: number
        durationSeconds: { min: number; max: number; avg: number }
        sampleRates: Record<string, number>
      } | null = null
      let text: {
        count: number
        tokenCount: { min: number; max: number; avg: number } | null
        languages: Record<string, number>
      } | null = null
      let tabular: { count: number } | null = null

      if (modality === 'vision') {
        const [agg] = await db
          .select({
            count: count(),
            minWidth: min(visionFeatures.width),
            maxWidth: max(visionFeatures.width),
            avgWidth: avg(visionFeatures.width),
            minHeight: min(visionFeatures.height),
            maxHeight: max(visionFeatures.height),
            avgHeight: avg(visionFeatures.height),
          })
          .from(visionFeatures)
          .innerJoin(datasetVersionItems, eq(visionFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
        const formatRows = await db
          .select({ format: visionFeatures.imageFormat, count: count() })
          .from(visionFeatures)
          .innerJoin(datasetVersionItems, eq(visionFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
          .groupBy(visionFeatures.imageFormat)
        vision = {
          count: agg.count,
          width: { min: agg.minWidth ?? 0, max: agg.maxWidth ?? 0, avg: Math.round(Number(agg.avgWidth ?? 0)) },
          height: { min: agg.minHeight ?? 0, max: agg.maxHeight ?? 0, avg: Math.round(Number(agg.avgHeight ?? 0)) },
          formats: Object.fromEntries(formatRows.map((r) => [r.format ?? 'unknown', r.count])),
        }
      }

      if (modality === 'audio') {
        const [agg] = await db
          .select({
            count: count(),
            minDuration: min(audioFeatures.durationSeconds),
            maxDuration: max(audioFeatures.durationSeconds),
            avgDuration: avg(audioFeatures.durationSeconds),
          })
          .from(audioFeatures)
          .innerJoin(datasetVersionItems, eq(audioFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
        const rateRows = await db
          .select({ rate: audioFeatures.sampleRateHz, count: count() })
          .from(audioFeatures)
          .innerJoin(datasetVersionItems, eq(audioFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
          .groupBy(audioFeatures.sampleRateHz)
        audio = {
          count: agg.count,
          durationSeconds: {
            min: Number(agg.minDuration ?? 0),
            max: Number(agg.maxDuration ?? 0),
            avg: Number(agg.avgDuration ?? 0),
          },
          sampleRates: Object.fromEntries(rateRows.map((r) => [String(r.rate), r.count])),
        }
      }

      if (modality === 'text') {
        const [agg] = await db
          .select({
            count: count(),
            minTokens: min(textFeatures.tokenCount),
            maxTokens: max(textFeatures.tokenCount),
            avgTokens: avg(textFeatures.tokenCount),
            withTokenCount: count(textFeatures.tokenCount),
          })
          .from(textFeatures)
          .innerJoin(datasetVersionItems, eq(textFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
        const langRows = await db
          .select({ language: textFeatures.languageCode, count: count() })
          .from(textFeatures)
          .innerJoin(datasetVersionItems, eq(textFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
          .groupBy(textFeatures.languageCode)
        text = {
          count: agg.count,
          tokenCount:
            agg.withTokenCount > 0
              ? { min: agg.minTokens ?? 0, max: agg.maxTokens ?? 0, avg: Math.round(Number(agg.avgTokens ?? 0)) }
              : null,
          languages: Object.fromEntries(langRows.map((r) => [r.language ?? 'unknown', r.count])),
        }
      }

      if (modality === 'tabular') {
        const [{ tabularCount }] = await db
          .select({ tabularCount: count() })
          .from(tabularFeatures)
          .innerJoin(datasetVersionItems, eq(tabularFeatures.itemId, datasetVersionItems.itemId))
          .where(eq(datasetVersionItems.versionId, draft.id))
        tabular = { count: tabularCount }
      }

      return {
        health: {
          itemCount,
          labeledCount,
          unlabeledCount: itemCount - labeledCount,
          modality,
          classDistribution,
          smallClasses,
          duplicateContentHashes: dupRows.length,
          missingContentHash,
          vision,
          audio,
          text,
          tabular,
        },
      }
    },
    { draftBelongToUser: true },
  )

  /* ================================================================== */
  /*  ANNOTATION ENDPOINTS                                              */
  /* ================================================================== */

  /* ── Add annotation to an item ── */
  .post(
    '/items/:itemId/annotations',
    async ({ item, body }) => {
      if (body.classId && !(await classInDataset(body.classId, item.dataset.projectId)))
        return status(400, 'Unknown label class')

      const [annotation] = await db
        .insert(annotations)
        .values({
          itemId: item.id,
          annotatorId: body.annotatorId,
          annotationType: body.annotationType,
          classId: body.classId,
          labelTextSequence: body.labelTextSequence,
          labelStructured: body.labelStructured,
          confidenceScore: body.confidenceScore != null ? String(body.confidenceScore) : undefined,
        })
        .returning()

      return { annotation }
    },
    {
      itemBelongToUser: true,
      body: t.Object({
        annotatorId: t.Optional(t.String()),
        annotationType: t.UnionEnum([
          'classification',
          'bounding_box',
          'segmentation_mask',
          'text_sequence',
          'token_tags',
          'preference_rank',
        ]),
        classId: t.Optional(t.String({ format: 'uuid' })),
        labelTextSequence: t.Optional(t.String()),
        labelStructured: t.Optional(t.Any()),
        confidenceScore: t.Optional(t.Number({ minimum: 0, maximum: 1 })),
      }),
    },
  )

  /* ── List annotations for an item ── */
  .get(
    '/items/:itemId/annotations',
    async ({ item }) => {
      const itemAnnotations = await db.query.annotations.findMany({ where: { itemId: item.id } })
      return { annotations: itemAnnotations }
    },
    { itemBelongToUser: true },
  )

  /* ── Update an annotation (re-labeling — avoids delete-then-create) ── */
  .patch(
    '/annotations/:annotationId',
    async ({ annotation, body }) => {
      if (body.classId && !(await classInDataset(body.classId, annotation.item.dataset.projectId)))
        return status(400, 'Unknown label class')

      const [updated] = await db
        .update(annotations)
        .set({
          classId: body.classId,
          labelTextSequence: body.labelTextSequence,
          labelStructured: body.labelStructured,
          confidenceScore: body.confidenceScore != null ? String(body.confidenceScore) : undefined,
        })
        .where(eq(annotations.id, annotation.id))
        .returning()

      return { annotation: updated }
    },
    {
      annotationBelongToUser: true,
      body: t.Object({
        classId: t.Optional(t.String({ format: 'uuid' })),
        labelTextSequence: t.Optional(t.String()),
        labelStructured: t.Optional(t.Any()),
        confidenceScore: t.Optional(t.Number({ minimum: 0, maximum: 1 })),
      }),
    },
  )

  /* ── Delete an annotation ── */
  .delete(
    '/annotations/:annotationId',
    async ({ annotation }) => {
      await db.delete(annotations).where(eq(annotations.id, annotation.id))
      return status(204)
    },
    { annotationBelongToUser: true },
  )

/** Maps a raw item-insert failure to a message safe to return to an API caller — see the call site above. */
function toSafeItemErrorMessage(reason: unknown): string {
  const cause = (reason as { cause?: { errno?: string } })?.cause
  if (cause?.errno === '23505') return 'An item with identical content already exists in this project'
  return 'Failed to create item'
}

/**
 * Deterministic sha256 of an inline (text/tabular) item's content, for the
 * same (datasetId, contentHash) dedup uploadToPool uses for files — object
 * key order must not affect the hash, so tabular features are serialized
 * with sorted keys rather than via a plain JSON.stringify.
 */
function hashItemContent(itemData: {
  textFeatures?: { rawText: string }
  tabularFeatures?: { featuresJson: unknown }
}): string | undefined {
  if (itemData.textFeatures) {
    return new Bun.CryptoHasher('sha256').update(itemData.textFeatures.rawText).digest('hex')
  }
  if (itemData.tabularFeatures) {
    return new Bun.CryptoHasher('sha256').update(canonicalJson(itemData.tabularFeatures.featuresJson)).digest('hex')
  }
  return undefined
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`
  }
  return JSON.stringify(value)
}

/**
 * Guards against attaching another tenant's label class to an item. The FK on
 * annotations.class_id only proves the class exists *somewhere* — a class from
 * a different dataset passes it, then resolves to `null` in buildSnapshot's
 * dataset-scoped classNameById map, silently producing unlabeled training rows.
 */
async function classInDataset(classId: string, datasetId: string): Promise<boolean> {
  const cls = await db.query.labelClasses.findFirst({
    where: { classId, datasetId },
    columns: { classId: true },
  })
  return cls != null
}

/**
 * Removes an item from the draft (mutable — always safe) then tries a hard
 * delete. If a *snapshot* still references it, the RESTRICT FK on
 * dataset_version_items blocks that — soft-delete instead so the row (and
 * its feature/annotation rows) stay intact for that snapshot while
 * disappearing from the pool, which the draft-membership removal above
 * already achieves on its own.
 */
async function deleteItemFromPool(
  itemId: string,
  datasetId: string,
): Promise<'deleted' | 'soft_deleted' | 'not_found'> {
  const draft = await db.query.datasetVersions.findFirst({
    where: { datasetId, versionTag: { isNull: true } },
  })
  if (draft) {
    await db
      .delete(datasetVersionItems)
      .where(and(eq(datasetVersionItems.versionId, draft.id), eq(datasetVersionItems.itemId, itemId)))
  }

  try {
    const [deleted] = await db
      .delete(datasetItems)
      .where(and(eq(datasetItems.id, itemId), eq(datasetItems.datasetId, datasetId)))
      .returning({ storageUrl: datasetItems.storageUrl })
    if (!deleted) return 'not_found'
    if (deleted.storageUrl) {
      await deleteFile(CONSTANTS.BUCKET_DATASETS, deleted.storageUrl).catch(() => {})
    }
    return 'deleted'
  } catch (e) {
    // PostgreSQL raises 23503 (foreign_key_violation) for an ON DELETE RESTRICT
    // FK. 23001 (restrict_violation) is in the errcode list but is not what
    // ri_triggers.c actually emits — accept both so the soft-delete path works.
    const cause = (e as { cause?: { errno?: string } })?.cause
    if (cause?.errno !== '23503' && cause?.errno !== '23001') throw e
    const [softDeleted] = await db
      .update(datasetItems)
      .set({ deletedAt: new Date() })
      .where(and(eq(datasetItems.id, itemId), eq(datasetItems.datasetId, datasetId)))
      .returning({ id: datasetItems.id })
    return softDeleted ? 'soft_deleted' : 'not_found'
  }
}

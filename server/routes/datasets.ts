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
  tabularFeatures,
  textFeatures,
  visionFeatures,
} from '@server/db/schema'
import { cleanupVersionStorage } from '@server/lib/cleanup'
import { type SplitType, SplitTypes } from '@server/lib/enums'
import { readImageDimensions } from '@server/lib/image-size'
import { buildSnapshot } from '@server/lib/snapshot'
import { deleteFile, getDownloadUrl, uploadToPool } from '@server/lib/storage'
import { and, countDistinct, eq } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

export const datasetRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)

  /* ================================================================== */
  /*  VERSION-LEVEL ENDPOINTS                                           */
  /* ================================================================== */

  /* ── Snapshot the draft's current pool membership into a new immutable version ── */
  .post(
    '/projects/:projectId/versions',
    async ({ project, body }) => {
      const draft = await db.query.datasetVersions.findFirst({
        where: { datasetId: project.id, versionTag: { isNull: true } },
      })
      if (!draft) return status(404, 'Draft dataset not found')

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
      projectBelongToUser: true,
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
    async ({ params, query }) => {
      const version = await db.query.datasetVersions.findFirst({
        where: query.versionId
          ? { id: query.versionId, datasetId: params.projectId }
          : { datasetId: params.projectId, versionTag: { isNull: true } },
      })
      if (!version) return status(404, 'Version not found')

      const page = Math.max(1, query.page ?? 1)
      const perPage = Math.min(200, Math.max(1, query.perPage ?? 30))
      const offset = (page - 1) * perPage

      // t.Optional(t.UnionEnum(...)) silently defaults to the first enum
      // value when the query param is absent instead of staying undefined,
      // so `split` is validated as a plain string and checked manually.
      const split: SplitType | undefined = (SplitTypes as readonly string[]).includes(query.split ?? '')
        ? (query.split as SplitType)
        : undefined
      const memberWhere = split ? { versionId: version.id, splitType: split } : { versionId: version.id }

      const [total, members] = await Promise.all([
        db.$count(datasetVersionItems, and(...whereClauses(memberWhere))),
        db.query.datasetVersionItems.findMany({
          where: memberWhere,
          with: {
            item: {
              with: {
                textFeatures: true,
                visionFeatures: true,
                audioFeatures: true,
                tabularFeatures: true,
                annotations: true,
              },
            },
          },
          limit: perPage,
          offset,
        }),
      ])

      const itemsWithUrls = await Promise.all(
        members.map(async (m) => ({
          ...m.item,
          splitType: m.splitType,
          downloadUrl: m.item.storageUrl
            ? await getDownloadUrl(CONSTANTS.BUCKET_DATASETS, m.item.storageUrl, 3600)
            : null,
        })),
      )

      return { items: itemsWithUrls, total, page, perPage }
    },
    {
      projectBelongToUser: true,
      query: t.Object({
        versionId: t.Optional(t.String()),
        split: t.Optional(t.String()),
        page: t.Optional(t.Numeric({ minimum: 1 })),
        perPage: t.Optional(t.Numeric({ minimum: 1, maximum: 200 })),
      }),
    },
  )

  /* ── Add items to the pool + draft, with a split assignment ── */
  .post(
    '/projects/:projectId/items',
    async ({ project, body }) => {
      const draft = await db.query.datasetVersions.findFirst({
        where: { datasetId: project.id, versionTag: { isNull: true } },
      })
      if (!draft) return status(404, 'Draft dataset not found')

      const results = await Promise.allSettled(
        body.items.map(async (itemData) => {
          const [item] = await db
            .insert(datasetItems)
            .values({
              datasetId: project.id,
              externalId: itemData.externalId,
              storageUrl: itemData.storageUrl,
            })
            .returning()

          if (itemData.textFeatures) {
            await db.insert(textFeatures).values({
              itemId: item.id,
              rawText: itemData.textFeatures.rawText,
              tokenCount: itemData.textFeatures.tokenCount,
              languageCode: itemData.textFeatures.languageCode,
              metaJson: itemData.textFeatures.metaJson,
            })
          }

          if (itemData.visionFeatures) {
            await db.insert(visionFeatures).values({
              itemId: item.id,
              width: itemData.visionFeatures.width,
              height: itemData.visionFeatures.height,
              channels: itemData.visionFeatures.channels,
              imageFormat: itemData.visionFeatures.imageFormat,
              exifData: itemData.visionFeatures.exifData,
            })
          }

          if (itemData.audioFeatures) {
            await db.insert(audioFeatures).values({
              itemId: item.id,
              durationSeconds: String(itemData.audioFeatures.durationSeconds),
              sampleRateHz: itemData.audioFeatures.sampleRateHz,
              channels: itemData.audioFeatures.channels,
              audioCodec: itemData.audioFeatures.audioCodec,
            })
          }

          if (itemData.tabularFeatures) {
            await db.insert(tabularFeatures).values({
              itemId: item.id,
              featuresJson: itemData.tabularFeatures.featuresJson,
            })
          }

          if (itemData.annotations?.length) {
            await db.insert(annotations).values(
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

          await db.insert(datasetVersionItems).values({
            versionId: draft.id,
            itemId: item.id,
            splitType: itemData.split,
          })

          return item
        }),
      )

      return { results }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        items: t.Array(
          t.Object({
            split: t.UnionEnum(SplitTypes),
            externalId: t.Optional(t.String()),
            storageUrl: t.Optional(t.String()),
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
                  classId: t.Optional(t.String()),
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
    async ({ project, body }) => {
      const draft = await db.query.datasetVersions.findFirst({
        where: { datasetId: project.id, versionTag: { isNull: true } },
      })
      if (!draft) return status(404, 'Draft dataset not found')

      const dataset = await db.query.datasets.findFirst({
        where: { projectId: project.id },
        columns: { modality: true },
      })
      if (!dataset) return status(404, 'Dataset not found')

      const results = await Promise.allSettled(
        body.files.map(async (file) => {
          const fileBytes = await file.bytes()
          const ext = file.name.includes('.') ? `.${file.name.split('.').pop()}` : ''
          const { key, hash, byteSize, isDuplicate } = await uploadToPool(project.id, fileBytes, ext, file.type)

          // The pool is content-addressed and deduplicated: if this exact
          // content already has an item, reuse it instead of violating the
          // (datasetId, contentHash) uniqueness constraint.
          const existingItem = await db.query.datasetItems.findFirst({
            where: { datasetId: project.id, contentHash: hash },
          })

          const item =
            existingItem ??
            (
              await db
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

          if (!existingItem && dataset.modality === 'vision' && file.type.startsWith('image/')) {
            const dimensions = readImageDimensions(fileBytes)
            if (dimensions) {
              await db.insert(visionFeatures).values({
                itemId: item.id,
                width: dimensions.width,
                height: dimensions.height,
                channels: 3,
                imageFormat: file.type === 'image/png' ? 'png' : 'jpeg',
              })
            }
          }

          await db
            .insert(datasetVersionItems)
            .values({
              versionId: draft.id,
              itemId: item.id,
              splitType: body.split,
            })
            .onConflictDoNothing()

          return { ...item, isDuplicate }
        }),
      )

      return { results }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        split: t.UnionEnum(SplitTypes),
        files: t.Files({ maxSize: '50m' }),
      }),
    },
  )

  /* ── Delete a single item from the pool ── */
  .delete(
    '/items/:itemId',
    async ({ item }) => {
      // Drop the item's membership in the draft (mutable — always safe to
      // remove) before attempting the delete. What's left is RESTRICT: any
      // *snapshot* still referencing the item blocks the delete, which is
      // exactly the immutability guarantee snapshots are meant to provide.
      const draft = await db.query.datasetVersions.findFirst({
        where: { datasetId: item.dataset.projectId, versionTag: { isNull: true } },
      })
      if (draft) {
        await db
          .delete(datasetVersionItems)
          .where(and(eq(datasetVersionItems.versionId, draft.id), eq(datasetVersionItems.itemId, item.id)))
      }

      try {
        await db.delete(datasetItems).where(eq(datasetItems.id, item.id))
      } catch (e) {
        const cause = (e as { cause?: { errno?: string } })?.cause
        if (cause?.errno === '23001') {
          return status(409, 'Cannot delete an item that belongs to a snapshot')
        }
        throw e
      }

      if (item.storageUrl) {
        await deleteFile(CONSTANTS.BUCKET_DATASETS, item.storageUrl).catch(() => {})
      }

      return status(204)
    },
    { itemBelongToUser: true },
  )

  /* ================================================================== */
  /*  ANNOTATION ENDPOINTS                                              */
  /* ================================================================== */

  /* ── Add annotation to an item ── */
  .post(
    '/items/:itemId/annotations',
    async ({ item, body }) => {
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
        classId: t.Optional(t.String()),
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
    async ({ params, user, body }) => {
      const annotation = await db.query.annotations.findFirst({
        where: { id: params.annotationId },
        with: { item: { with: { dataset: true } } },
      })
      if (!annotation) return status(404, 'Annotation not found')

      const project = await db.query.projects.findFirst({
        where: { id: annotation.item.dataset.projectId },
        columns: { userId: true },
      })
      if (!project || project.userId !== user.id) return status(403, 'Unauthorized')

      const [updated] = await db
        .update(annotations)
        .set({
          classId: body.classId,
          labelTextSequence: body.labelTextSequence,
          labelStructured: body.labelStructured,
          confidenceScore: body.confidenceScore != null ? String(body.confidenceScore) : undefined,
        })
        .where(eq(annotations.id, params.annotationId))
        .returning()

      return { annotation: updated }
    },
    {
      auth: true,
      body: t.Object({
        classId: t.Optional(t.String()),
        labelTextSequence: t.Optional(t.String()),
        labelStructured: t.Optional(t.Any()),
        confidenceScore: t.Optional(t.Number({ minimum: 0, maximum: 1 })),
      }),
    },
  )

  /* ── Delete an annotation ── */
  .delete(
    '/annotations/:annotationId',
    async ({ params, user }) => {
      const annotation = await db.query.annotations.findFirst({
        where: { id: params.annotationId },
        with: {
          item: {
            with: { dataset: true },
          },
        },
      })
      if (!annotation) return status(404, 'Annotation not found')

      const project = await db.query.projects.findFirst({
        where: { id: annotation.item.dataset.projectId },
        columns: { userId: true },
      })
      if (!project || project.userId !== user.id) return status(403, 'Unauthorized')

      await db.delete(annotations).where(eq(annotations.id, params.annotationId))
      return status(204)
    },
    { auth: true },
  )

/** Build a Drizzle SQL where-clause array from a simple relational filter object. */
function whereClauses(filter: Record<string, unknown>) {
  const clauses = []
  if (filter.versionId) clauses.push(eq(datasetVersionItems.versionId, filter.versionId as string))
  if (filter.splitType) clauses.push(eq(datasetVersionItems.splitType, filter.splitType as (typeof SplitTypes)[number]))
  return clauses
}

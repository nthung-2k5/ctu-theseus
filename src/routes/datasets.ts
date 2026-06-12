import { db } from '@server/db'
import { datasetImages } from '@server/db/schema'
import { BUCKET_DATASETS, deleteFile, downloadFile, fileExists, getDownloadUrl, uploadFile } from '@server/lib/storage'
import { and, asc, desc, eq, ilike, inArray } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

export const datasetRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)
  /* ── List images (paginated, filtered, sorted) ── */
  .get(
    '/projects/:projectId/images',
    async ({ params, query }) => {
      const page = Math.max(1, query.page ?? 1)
      const perPage = Math.min(200, Math.max(1, query.perPage ?? 30))
      const offset = (page - 1) * perPage

      // Build WHERE conditions
      const conditions = [eq(datasetImages.projectId, params.projectId)]

      if (query.search) {
        conditions.push(ilike(datasetImages.filename, `%${query.search}%`))
      }

      if (query.classId !== undefined) {
        conditions.push(eq(datasetImages.classId, query.classId))
      }

      if (query.split !== undefined) {
        conditions.push(eq(datasetImages.split, query.split))
      }

      const whereClause = and(...conditions)

      // Build ORDER BY
      let orderBy: ReturnType<typeof asc> | ReturnType<typeof desc> | undefined
      switch (query.sort) {
        case 'newest':
          orderBy = desc(datasetImages.uploadedAt)
          break
        case 'oldest':
          orderBy = asc(datasetImages.uploadedAt)
          break
        case 'filename':
          orderBy = asc(datasetImages.filename)
          break
        default:
          orderBy = desc(datasetImages.uploadedAt) // default sort
      }

      // Execute count and paginated query in parallel
      const [total, images] = await Promise.all([
        db.$count(datasetImages, whereClause),
        db.select().from(datasetImages).where(whereClause).orderBy(orderBy).limit(perPage).offset(offset),
      ])

      // Generate presigned S3 URLs for each image
      const imagesWithUrls = await Promise.all(
        images.map(async (img) => ({
          ...img,
          url: await getDownloadUrl(BUCKET_DATASETS, img.path, 3600),
        })),
      )

      return {
        images: imagesWithUrls,
        total,
        page,
        perPage,
      }
    },
    {
      projectBelongToUser: true,
      query: t.Object({
        page: t.Optional(t.Numeric({ minimum: 1 })),
        perPage: t.Optional(t.Numeric({ minimum: 1, maximum: 200 })),
        search: t.Optional(t.String()),
        classId: t.Optional(t.String()),
        split: t.Optional(t.UnionEnum(['train', 'validation', 'test'], { default: undefined })),
        sort: t.Optional(t.UnionEnum(['none', 'newest', 'oldest', 'filename'], { default: undefined })),
      }),
    },
  )
  /* ── Stats (split counts — lightweight, no pagination) ── */
  .get(
    '/projects/:projectId/images/stats',
    async ({ params }) => {
      const images = await db.query.datasetImages.findMany({
        where: { projectId: params.projectId },
        columns: { split: true },
      })

      const counts = { train: 0, validation: 0, test: 0, unassigned: 0, total: images.length }
      for (const img of images) {
        if (img.split === 'train') counts.train++
        else if (img.split === 'validation') counts.validation++
        else if (img.split === 'test') counts.test++
        else counts.unassigned++
      }

      return counts
    },
    {
      projectBelongToUser: true,
    },
  )
  /* ── Upload single image ── */
  .post(
    '/projects/:projectId/images',
    async ({ params, body }) => {
      const images: [File, string | null][] = 'image' in body ? [[body.image, body.classId]] : body.images

      const result = await Promise.allSettled(
        images.map(async ([img, classId]) => {
          const originalName = img.name
          let finalName = originalName

          // Check for existing file with the same name in S3
          const s3Key = `images/${params.projectId}/${originalName}`
          if (await fileExists(BUCKET_DATASETS, s3Key)) {
            // Compare content via hash
            const existingBytes = await downloadFile(BUCKET_DATASETS, s3Key)
            const existingHash = new Bun.CryptoHasher('md5').update(existingBytes).digest('hex')
            const newHash = new Bun.CryptoHasher('md5').update(await img.arrayBuffer()).digest('hex')

            if (existingHash === newHash) {
              // Identical content — find existing DB record and return it
              const existing = await db.query.datasetImages.findFirst({
                where: { projectId: params.projectId, filename: originalName },
              })
              if (existing) {
                return existing
              }
              // If not exists then create a new record.
            } else {
              // Different content — find a unique suffix
              const suffix = Bun.randomUUIDv7()
              finalName = `${originalName}_${suffix}`
            }
          }

          const targetKey = `images/${params.projectId}/${finalName}`

          // Upload to S3
          const buffer = new Uint8Array(await img.arrayBuffer())
          await uploadFile(BUCKET_DATASETS, targetKey, buffer, img.type)

          // Read image dimensions
          const imgObj = img.image()
          const width = imgObj.width
          const height = imgObj.height

          // Insert into database (path = S3 key)
          const [row] = await db
            .insert(datasetImages)
            .values({
              projectId: params.projectId,
              filename: finalName,
              classId: classId || null,
              path: targetKey,
              width,
              height,
            })
            .returning()

          return row
        }),
      )

      return {
        images: result,
      }
    },
    {
      projectBelongToUser: true,
      body: t.Union([
        t.Object({
          image: t.File({ type: 'image/*', maxSize: '5m' }),
          classId: t.Nullable(t.String()),
        }),
        t.Object({
          images: t.Array(t.Tuple([t.File({ type: 'image/*', maxSize: '5m' }), t.Nullable(t.String())])),
        }),
      ]),
    },
  )
  /* ── Update image (class, split) ── */
  .patch(
    '/images/:imageId',
    async ({ params, body, user }) => {
      const image = await db.query.datasetImages.findFirst({
        where: { id: params.imageId },
        with: {
          projects: {
            where: {
              userId: user.id,
            },
          },
        },
      })

      if (!image) throw new Error('Image not found')

      const [updatedImage] = await db
        .update(datasetImages)
        .set(body)
        .where(eq(datasetImages.id, params.imageId))
        .returning()

      return { image: updatedImage }
    },
    {
      auth: true,
      body: t.Partial(
        t.Object({
          classId: t.Nullable(t.String()),
          split: t.Nullable(t.UnionEnum(['train', 'validation', 'test'], { default: undefined })),
        }),
      ),
    },
  )
  /* ── Batch assign splits ── */
  .post(
    '/projects/:projectId/images/auto-split',
    async ({ params, body }) => {
      if (body.train + body.validation + body.test !== 100) {
        return status(400, 'Percentages must sum to 100')
      }

      const images = await db.query.datasetImages.findMany({
        where: { projectId: params.projectId },
      })

      if (images.length === 0) return { counts: { train: 0, validation: 0, test: 0 } }

      // Shuffle
      const shuffled = [...images].sort(() => Math.random() - 0.5)
      const trainEnd = Math.round(shuffled.length * (body.train / 100))
      const valEnd = trainEnd + Math.round(shuffled.length * (body.validation / 100))

      const trainIds = shuffled.slice(0, trainEnd).map((i) => i.id)
      const valIds = shuffled.slice(trainEnd, valEnd).map((i) => i.id)
      const testIds = shuffled.slice(valEnd).map((i) => i.id)

      await Promise.all([
        trainIds.length > 0 &&
          db.update(datasetImages).set({ split: 'train' }).where(inArray(datasetImages.id, trainIds)),
        valIds.length > 0 &&
          db.update(datasetImages).set({ split: 'validation' }).where(inArray(datasetImages.id, valIds)),
        testIds.length > 0 && db.update(datasetImages).set({ split: 'test' }).where(inArray(datasetImages.id, testIds)),
      ])

      return {
        counts: { train: trainIds.length, validation: valIds.length, test: testIds.length },
      }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        train: t.Number({ minimum: 0, maximum: 100 }),
        validation: t.Number({ minimum: 0, maximum: 100 }),
        test: t.Number({ minimum: 0, maximum: 100 }),
      }),
    },
  )
  /* ── Delete image ── */
  .delete(
    '/images/:imageId',
    async ({ params, user }) => {
      const image = await db.query.datasetImages.findFirst({
        where: { id: params.imageId },
        with: {
          projects: {
            where: {
              userId: user.id,
            },
          },
        },
      })

      if (!image) throw new Error('Image not found')

      await Promise.all([
        db.delete(datasetImages).where(eq(datasetImages.id, params.imageId)),
        deleteFile(BUCKET_DATASETS, image.path),
      ])

      return status(204)
    },
    {
      auth: true,
      params: t.Object({
        imageId: t.String(),
      }),
    },
  )
  /* ── Get presigned URL for viewing an image ── */
  .get(
    '/images/:imageId/url',
    async ({ params, user }) => {
      const image = await db.query.datasetImages.findFirst({
        where: { id: params.imageId },
        with: {
          projects: {
            where: {
              userId: user.id,
            },
          },
        },
      })

      if (!image) throw new Error('Image not found')

      const url = await getDownloadUrl(BUCKET_DATASETS, image.path, 3600)
      return { url }
    },
    {
      auth: true,
      params: t.Object({
        imageId: t.String(),
      }),
    },
  )

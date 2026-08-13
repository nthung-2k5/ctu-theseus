/**
 * Label Classes routes – CRUD for classification labels.
 *
 * Label classes belong to a dataset (1:1 with project), so the routes
 * are scoped under /api/projects/:projectId/classes.
 *
 * Only relevant for classification tasks:
 *   text_classification, image_classification, audio_classification, tabular_classification
 */

import { db } from '@server/db'
import { labelClasses } from '@server/db/schema'
import { eq } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/** Default palette for auto-assigning colors to new classes */
const CLASS_COLORS = [
  '#e03131',
  '#2f9e44',
  '#1971c2',
  '#f08c00',
  '#9c36b5',
  '#0c8599',
  '#e8590c',
  '#6741d9',
  '#3bc9db',
  '#ff6b6b',
  '#51cf66',
  '#339af0',
  '#fcc419',
  '#cc5de8',
  '#20c997',
  '#ff922b',
]

export const classRoutes = new Elysia({ prefix: '/api/projects/:projectId/classes' })
  .use(betterAuth)

  /* ── List all label classes for a project's dataset ── */
  .get(
    '/',
    async ({ project }) => {
      // datasetId === projectId (1:1)
      const classes = await db.query.labelClasses.findMany({
        where: { datasetId: project.id, isActive: true },
        orderBy: { createdAt: 'asc' },
      })
      return { classes }
    },
    { projectBelongToUser: true },
  )

  /* ── Create a new label class ── */
  .post(
    '/',
    async ({ project, body }) => {
      // Auto-assign color if not provided
      let color = body.uiColorHex
      if (!color) {
        const existingCount = await db.$count(labelClasses, eq(labelClasses.datasetId, project.id))
        color = CLASS_COLORS[existingCount % CLASS_COLORS.length]
      }

      // Check if class name already exists
      const existingClass = await db.query.labelClasses.findFirst({
        where: { datasetId: project.id, name: body.name, isActive: true },
      })

      if (existingClass) {
        return status(400, 'Class name already exists')
      }

      const [cls] = await db
        .insert(labelClasses)
        .values({
          datasetId: project.id,
          name: body.name,
          description: body.description,
          uiColorHex: color,
        })
        .returning()

      return { class: cls }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        name: t.String({ minLength: 1, maxLength: 100 }),
        description: t.Optional(t.String()),
        uiColorHex: t.Optional(t.String({ pattern: '^#[0-9a-fA-F]{6}$' })),
      }),
    },
  )

  /* ── Update a label class ── */
  .patch(
    '/:classId',
    async ({ project, params, body }) => {
      const cls = await db.query.labelClasses.findFirst({
        where: { classId: params.classId, isActive: true },
      })
      if (!cls) return status(404, 'Class not found')
      if (cls.datasetId !== project.id) return status(403, 'Unauthorized')

      const [updated] = await db
        .update(labelClasses)
        .set(body)
        .where(eq(labelClasses.classId, params.classId))
        .returning()

      return { class: updated }
    },
    {
      projectBelongToUser: true,
      // projectBelongToUser's own `params` schema only declares `projectId`;
      // without redeclaring the full set here, Elysia validates against that
      // narrower shape and rejects `classId` as an unexpected property
      // before the handler (or the macro's ownership check) ever runs.
      params: t.Object({ projectId: t.String(), classId: t.String() }),
      body: t.Object({
        name: t.Optional(t.String({ minLength: 1, maxLength: 100 })),
        description: t.Optional(t.String()),
        uiColorHex: t.Optional(t.String({ pattern: '^#[0-9a-fA-F]{6}$' })),
      }),
    },
  )

  /* ── Delete a label class ── */
  .delete(
    '/:classId',
    async ({ project, params }) => {
      const cls = await db.query.labelClasses.findFirst({
        where: { classId: params.classId },
      })
      if (!cls) return status(404, 'Class not found')
      if (cls.datasetId !== project.id) return status(403, 'Unauthorized')

      // Note: annotations referencing this class use ON DELETE RESTRICT,
      // so deletion will fail if annotations exist. The client should
      // handle this gracefully.
      try {
        await db.update(labelClasses).set({ isActive: false }).where(eq(labelClasses.classId, params.classId))
        return status(204)
      } catch {
        return status(409, 'Cannot delete class: it is referenced by existing annotations')
      }
    },
    { projectBelongToUser: true, params: t.Object({ projectId: t.String(), classId: t.String() }) },
  )

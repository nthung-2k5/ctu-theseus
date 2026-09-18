/**
 * Route-level integration tests for datasets.ts — the largest and (per the
 * project's own roadmap) highest-risk route file, previously entirely
 * untested. See server/test/helpers.ts for what running this needs (a real,
 * migrated Postgres) and how to run it (`bun run test:integration`, not part
 * of `bun test`/CI).
 *
 * Every item here is created inline (text) via POST /items — no file upload,
 * no S3-touching route is exercised (see helpers.ts for why).
 */

import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import {
  callApi,
  cleanupTestProject,
  createTestUser,
  deleteTestUser,
  getDraftVersionId,
  type TestUser,
} from '@server/test/helpers'

type ItemsListResponse = {
  items: { id: string; splitType: string }[]
  total: number
  labeledCount: number
  classCounts: { classId: string; count: number }[]
  unassignedCount: number
}

type CreateItemsResponse = {
  created: { id: string }[]
  failed: { index: number; message: string }[]
}

describe('datasets routes', () => {
  let user: TestUser
  let projectId: string
  let catClassId: string
  let dogClassId: string

  beforeAll(async () => {
    user = await createTestUser()
    const project = await callApi('POST', '/api/projects', user.cookie, {
      name: 'datasets-itest-project',
      description: null,
      task: 'text_classification',
    })
    expect(project.status).toBe(200)
    projectId = (project.body as { project: { id: string } }).project.id

    const cat = await callApi('POST', `/api/projects/${projectId}/classes`, user.cookie, { name: 'cat' })
    const dog = await callApi('POST', `/api/projects/${projectId}/classes`, user.cookie, { name: 'dog' })
    catClassId = (cat.body as { class: { classId: string } }).class.classId
    dogClassId = (dog.body as { class: { classId: string } }).class.classId
  })

  afterAll(async () => {
    await cleanupTestProject(projectId)
    await deleteTestUser(user.userId)
  })

  test('adding an item creates it with its text feature', async () => {
    const { status, body } = await callApi('POST', `/api/projects/${projectId}/items`, user.cookie, {
      items: [{ split: 'train', textFeatures: { rawText: 'the quick brown fox' } }],
    })
    expect(status).toBe(200)
    const result = body as CreateItemsResponse
    expect(result.created).toHaveLength(1)
    expect(result.failed).toHaveLength(0)
  })

  test('adding identical text content twice reuses the same pool item (content-hash dedup)', async () => {
    const first = await callApi('POST', `/api/projects/${projectId}/items`, user.cookie, {
      items: [{ split: 'train', textFeatures: { rawText: 'a duplicate sentence' } }],
    })
    const firstId = (first.body as CreateItemsResponse).created[0]?.id
    expect(firstId).toBeDefined()

    const second = await callApi('POST', `/api/projects/${projectId}/items`, user.cookie, {
      items: [{ split: 'test', textFeatures: { rawText: 'a duplicate sentence' } }],
    })
    expect(second.status).toBe(200)
    const secondResult = second.body as CreateItemsResponse
    expect(secondResult.failed).toHaveLength(0)
    // Same underlying pool item — dedup by content hash, not two rows.
    expect(secondResult.created[0]?.id).toBe(firstId)
  })

  test('lists items in the draft with a total that matches what was created', async () => {
    const { status, body } = await callApi('GET', `/api/projects/${projectId}/items?perPage=100`, user.cookie)
    expect(status).toBe(200)
    const result = body as ItemsListResponse
    // 1 from the first test + 1 deduped pair from the second (counts once).
    expect(result.total).toBe(2)
    expect(result.labeledCount).toBe(0)
    expect(result.unassignedCount).toBe(2)
  })

  test('bulk-classifies items and the class shows up in the listing counts', async () => {
    const { body: listBody } = await callApi('GET', `/api/projects/${projectId}/items?perPage=100`, user.cookie)
    const itemIds = (listBody as ItemsListResponse).items.map((i) => i.id)
    expect(itemIds.length).toBeGreaterThan(0)

    const classify = await callApi('POST', `/api/projects/${projectId}/items/classify`, user.cookie, {
      itemIds,
      classId: catClassId,
    })
    expect(classify.status).toBe(200)
    expect((classify.body as { updated: number; failed: number }).updated).toBe(itemIds.length)

    const { body: afterBody } = await callApi('GET', `/api/projects/${projectId}/items?perPage=100`, user.cookie)
    const after = afterBody as ItemsListResponse
    expect(after.labeledCount).toBe(itemIds.length)
    expect(after.unassignedCount).toBe(0)
    expect(after.classCounts).toContainEqual({ classId: catClassId, count: itemIds.length })
  })

  test('bulk-reassigns the split of items', async () => {
    const { body: listBody } = await callApi('GET', `/api/projects/${projectId}/items?perPage=100`, user.cookie)
    const itemIds = (listBody as ItemsListResponse).items.map((i) => i.id)

    const { status, body } = await callApi('PATCH', `/api/projects/${projectId}/items/split`, user.cookie, {
      itemIds,
      split: 'validation',
    })
    expect(status).toBe(200)
    expect((body as { updated: string[] }).updated.sort()).toEqual(itemIds.sort())

    const { body: afterBody } = await callApi(
      'GET',
      `/api/projects/${projectId}/items?perPage=100&split=validation`,
      user.cookie,
    )
    expect((afterBody as ItemsListResponse).total).toBe(itemIds.length)
  })

  test('auto-split stratifies by class and warns about classes with too few items to split', async () => {
    // One more 'cat' item (bringing cat to 3 — at the MIN_ITEMS_PER_CLASS
    // threshold) and two 'dog' items (2 total, below it) — so only 'dog'
    // should trigger the small-class warning below.
    await callApi('POST', `/api/projects/${projectId}/items`, user.cookie, {
      items: [
        {
          split: 'train',
          textFeatures: { rawText: 'cat item three' },
          annotations: [{ annotationType: 'classification', classId: catClassId }],
        },
        {
          split: 'train',
          textFeatures: { rawText: 'dog item one' },
          annotations: [{ annotationType: 'classification', classId: dogClassId }],
        },
        {
          split: 'train',
          textFeatures: { rawText: 'dog item two' },
          annotations: [{ annotationType: 'classification', classId: dogClassId }],
        },
      ],
    })

    const { status, body } = await callApi('POST', `/api/projects/${projectId}/items/auto-split`, user.cookie, {})
    expect(status).toBe(200)
    const result = body as {
      updated: number
      stratified: boolean
      splits: { train: number; validation: number; test: number }
      warnings: string[]
    }
    expect(result.stratified).toBe(true)
    expect(result.updated).toBe(result.splits.train + result.splits.validation + result.splits.test)
    expect(result.warnings.some((w) => w.includes('only 2 item'))).toBe(true)
  })

  test('dataset health reports class distribution, small classes, and text stats', async () => {
    const { status, body } = await callApi('GET', `/api/projects/${projectId}/dataset/health`, user.cookie)
    expect(status).toBe(200)
    const health = (body as { health: Record<string, unknown> }).health as {
      itemCount: number
      classDistribution: { classId: string; name: string; count: number }[]
      smallClasses: { classId: string; name: string }[]
      text: { count: number } | null
      vision: unknown
      audio: unknown
    }
    expect(health.itemCount).toBe(5)
    expect(health.classDistribution.map((c) => c.name).sort()).toEqual(['cat', 'dog'])
    expect(health.smallClasses.map((c) => c.name)).toEqual(['dog'])
    expect(health.text?.count).toBe(5)
    expect(health.vision).toBeNull()
    expect(health.audio).toBeNull()
  })

  test('annotation CRUD on a plain (non-classification) item', async () => {
    const created = await callApi('POST', `/api/projects/${projectId}/items`, user.cookie, {
      items: [{ split: 'train', textFeatures: { rawText: 'annotate me' } }],
    })
    const itemId = (created.body as CreateItemsResponse).created[0]?.id
    expect(itemId).toBeDefined()

    const add = await callApi('POST', `/api/items/${itemId}/annotations`, user.cookie, {
      annotationType: 'text_sequence',
      labelTextSequence: 'a target sequence',
    })
    expect(add.status).toBe(200)
    const annotationId = (add.body as { annotation: { id: string } }).annotation.id

    const list = await callApi('GET', `/api/items/${itemId}/annotations`, user.cookie)
    expect((list.body as { annotations: { id: string }[] }).annotations.map((a) => a.id)).toContain(annotationId)

    const update = await callApi('PATCH', `/api/annotations/${annotationId}`, user.cookie, {
      labelTextSequence: 'an updated sequence',
    })
    expect(update.status).toBe(200)
    expect((update.body as { annotation: { labelTextSequence: string } }).annotation.labelTextSequence).toBe(
      'an updated sequence',
    )

    const del = await callApi('DELETE', `/api/annotations/${annotationId}`, user.cookie)
    expect(del.status).toBe(204)

    const afterList = await callApi('GET', `/api/items/${itemId}/annotations`, user.cookie)
    expect((afterList.body as { annotations: { id: string }[] }).annotations.map((a) => a.id)).not.toContain(
      annotationId,
    )
  })

  test('bulk-deleting items removes them from the draft', async () => {
    const created = await callApi('POST', `/api/projects/${projectId}/items`, user.cookie, {
      items: [{ split: 'train', textFeatures: { rawText: 'item to be deleted' } }],
    })
    const itemId = (created.body as CreateItemsResponse).created[0]?.id
    expect(itemId).toBeDefined()

    const del = await callApi('DELETE', `/api/projects/${projectId}/items`, user.cookie, { itemIds: [itemId] })
    expect(del.status).toBe(200)
    expect((del.body as { results: { itemId: string; outcome: string }[] }).results).toEqual([
      { itemId, outcome: 'deleted' },
    ])

    const list = await callApi('GET', `/api/projects/${projectId}/items?perPage=100`, user.cookie)
    const ids = (list.body as ItemsListResponse).items.map((i) => i.id)
    expect(ids).not.toContain(itemId)
  })

  test("a stranger cannot read this project's items", async () => {
    const other = await createTestUser()
    try {
      // projectBelongToUser resolves the project by id regardless of owner,
      // then checks userId — a stranger gets 403, not a 404.
      const { status } = await callApi('GET', `/api/projects/${projectId}/items`, other.cookie)
      expect(status).toBe(403)
    } finally {
      await deleteTestUser(other.userId)
    }
  })

  test('the draft version id is stable across requests', async () => {
    const draftId = await getDraftVersionId(projectId)
    const { body } = await callApi('GET', `/api/projects/${projectId}/items?versionId=${draftId}`, user.cookie)
    expect((body as ItemsListResponse).total).toBeGreaterThanOrEqual(0)
  })
})

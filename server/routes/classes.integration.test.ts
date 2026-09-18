/**
 * Route-level integration tests for classes.ts — see server/test/helpers.ts
 * for what this needs (a real, migrated Postgres) and how to run it
 * (`bun run test:integration`, not part of `bun test`/CI).
 */

import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { callApi, cleanupTestProject, createTestUser, deleteTestUser, type TestUser } from '@server/test/helpers'

describe('classes routes', () => {
  let user: TestUser
  let projectId: string

  beforeAll(async () => {
    user = await createTestUser()
    const { status, body } = await callApi('POST', '/api/projects', user.cookie, {
      name: 'classes-itest-project',
      description: null,
      task: 'text_classification',
    })
    expect(status).toBe(200)
    projectId = (body as { project: { id: string } }).project.id
  })

  afterAll(async () => {
    await cleanupTestProject(projectId)
    await deleteTestUser(user.userId)
  })

  test('starts with no classes', async () => {
    const { status, body } = await callApi('GET', `/api/projects/${projectId}/classes`, user.cookie)
    expect(status).toBe(200)
    expect((body as { classes: unknown[] }).classes).toEqual([])
  })

  test('creates a class with an auto-assigned color', async () => {
    const { status, body } = await callApi('POST', `/api/projects/${projectId}/classes`, user.cookie, {
      name: 'cat',
    })
    expect(status).toBe(200)
    const cls = (body as { class: { classId: string; name: string; uiColorHex: string } }).class
    expect(cls.name).toBe('cat')
    expect(cls.uiColorHex).toMatch(/^#[0-9a-fA-F]{6}$/)
  })

  test('rejects a duplicate class name', async () => {
    await callApi('POST', `/api/projects/${projectId}/classes`, user.cookie, { name: 'dog' })
    const { status } = await callApi('POST', `/api/projects/${projectId}/classes`, user.cookie, { name: 'dog' })
    expect(status).toBe(400)
  })

  test('lists every active class', async () => {
    const { status, body } = await callApi('GET', `/api/projects/${projectId}/classes`, user.cookie)
    expect(status).toBe(200)
    const names = (body as { classes: { name: string }[] }).classes.map((c) => c.name).sort()
    expect(names).toEqual(['cat', 'dog'])
  })

  test('updates a class', async () => {
    const { body: listBody } = await callApi('GET', `/api/projects/${projectId}/classes`, user.cookie)
    const cat = (listBody as { classes: { classId: string; name: string }[] }).classes.find((c) => c.name === 'cat')
    expect(cat).toBeDefined()

    const { status, body } = await callApi('PATCH', `/api/projects/${projectId}/classes/${cat?.classId}`, user.cookie, {
      description: 'a feline',
    })
    expect(status).toBe(200)
    expect((body as { class: { description: string | null } }).class.description).toBe('a feline')
  })

  test('soft-deletes a class and it disappears from the active list', async () => {
    const { body: listBody } = await callApi('GET', `/api/projects/${projectId}/classes`, user.cookie)
    const dog = (listBody as { classes: { classId: string; name: string }[] }).classes.find((c) => c.name === 'dog')
    expect(dog).toBeDefined()

    const del = await callApi('DELETE', `/api/projects/${projectId}/classes/${dog?.classId}`, user.cookie)
    expect(del.status).toBe(204)

    const { body: afterBody } = await callApi('GET', `/api/projects/${projectId}/classes`, user.cookie)
    const names = (afterBody as { classes: { name: string }[] }).classes.map((c) => c.name)
    expect(names).not.toContain('dog')
  })

  test("another user cannot see or modify this project's classes", async () => {
    const other = await createTestUser()
    try {
      const list = await callApi('GET', `/api/projects/${projectId}/classes`, other.cookie)
      // projectBelongToUser resolves the project by id regardless of owner,
      // then checks userId — a stranger gets 403, not a filtered empty list.
      expect(list.status).toBe(403)
    } finally {
      await deleteTestUser(other.userId)
    }
  })
})

import { expect, test } from 'bun:test'
import type { ProjectSummary } from '@public/lib/api/generated/models'
import { filterProjects, nextStep } from './projectsList'

const project = (over: Partial<ProjectSummary>): ProjectSummary => ({
  id: 'p',
  name: 'Project',
  description: null,
  task: 'image_classification',
  createdAt: '2026-01-01T00:00:00Z',
  itemCount: 0,
  snapshotCount: 0,
  runCount: 0,
  ...over,
})

const modalityOf = (task: string) => (task.startsWith('text') ? 'text' : 'vision')
const all = { q: '', modality: null, sort: 'recent' as const }

test('the next step follows the order work flows', () => {
  expect(nextStep(project({})).kind).toBe('data')
  expect(nextStep(project({ itemCount: 10 })).kind).toBe('snapshot')
  expect(nextStep(project({ itemCount: 10, snapshotCount: 1 })).kind).toBe('train')
  expect(nextStep(project({ itemCount: 10, snapshotCount: 1, runCount: 2 })).kind).toBe('runs')
})

test('a run in progress comes before everything else and carries its id', () => {
  const step = nextStep(project({ activeRun: { id: 'r1', name: 'x', status: 'running' } }))
  expect(step).toEqual({ kind: 'run', label: 'View run', runId: 'r1' })
})

test('recent sorts by last activity, not by when the project was created', () => {
  const old = project({ id: 'old', createdAt: '2026-01-01T00:00:00Z', lastActivityAt: '2026-03-01T00:00:00Z' })
  const fresh = project({ id: 'fresh', createdAt: '2026-02-01T00:00:00Z', lastActivityAt: '2026-02-01T00:00:00Z' })
  expect(filterProjects([fresh, old], all, modalityOf).map((p) => p.id)).toEqual(['old', 'fresh'])
  expect(filterProjects([old, fresh], { ...all, sort: 'created' }, modalityOf).map((p) => p.id)).toEqual([
    'fresh',
    'old',
  ])
})

test('name sorts case-insensitively', () => {
  const names = ['beta', 'Alpha', 'gamma'].map((name) => project({ id: name, name }))
  expect(filterProjects(names, { ...all, sort: 'name' }, modalityOf).map((p) => p.name)).toEqual([
    'Alpha',
    'beta',
    'gamma',
  ])
})

test('search matches the name or the description, and a modality filter narrows by task', () => {
  const a = project({ id: 'a', name: 'Cats', task: 'image_classification' })
  const b = project({ id: 'b', name: 'Tickets', task: 'text_classification', description: 'support queue' })
  expect(filterProjects([a, b], { ...all, q: 'queue' }, modalityOf).map((p) => p.id)).toEqual(['b'])
  expect(filterProjects([a, b], { ...all, q: 'CAT' }, modalityOf).map((p) => p.id)).toEqual(['a'])
  expect(filterProjects([a, b], { ...all, modality: 'text' }, modalityOf).map((p) => p.id)).toEqual(['b'])
})

test('filtering does not reorder or mutate the list it was given', () => {
  const list = [project({ id: 'x', name: 'b' }), project({ id: 'y', name: 'a' })]
  filterProjects(list, { ...all, sort: 'name' }, modalityOf)
  expect(list.map((p) => p.id)).toEqual(['x', 'y'])
})

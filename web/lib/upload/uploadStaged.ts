/**
 * Sends the staged files to the server.
 *
 * `POST /projects/:id/upload` takes one split + class per request, so files are grouped by that pair and each
 * group is cut into bounded chunks that go out a few at a time. Results are reported chunk by chunk: files the
 * server took leave the tree immediately, and files it refused stay with the reason attached so a retry only
 * resends those. Classes for folders that don't exist yet are created first.
 */

import { createClass, listClasses } from '@public/lib/api/generated/classes/classes'
import { uploadItems } from '@public/lib/api/generated/datasets/datasets'
import { isAxiosError } from 'axios'
import { classIdOfKey, isPendingKey } from './fileTree'
import { normalizeClassName } from './paths'
import type { PendingFolder, StagedFile } from './types'

const MAX_CHUNK_FILES = 50
const MAX_CHUNK_BYTES = 64 * 1024 * 1024
const CONCURRENCY = 3

export interface UploadProgress {
  /** Files settled so far, uploaded or failed. */
  settled: number
  total: number
}

export interface UploadOutcome {
  uploaded: number
  /** Uploaded files whose content the project already had (the server deduplicated them). */
  duplicates: number
  failed: number
  createdClasses: number
  cancelled: boolean
}

export interface UploadRequest {
  projectId: string
  /** Only files that passed validation. */
  files: StagedFile[]
  pending: Map<string, PendingFolder>
  signal: AbortSignal
  onProgress: (progress: UploadProgress) => void
  /** Called after each chunk: ids the server took, and reasons for the ones it refused. */
  onSettled: (uploadedIds: string[], errors: Record<string, string>) => void
}

/** The server's `detail` when it sent one, else the transport error. */
function describeError(error: unknown, fallback: string): string {
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (error.response) return `${fallback} (HTTP ${error.response.status})`
    return error.message
  }
  return error instanceof Error ? error.message : fallback
}

interface Chunk {
  split: string
  classId: string | null
  files: StagedFile[]
}

function toChunks(files: StagedFile[], classIdOf: (file: StagedFile) => string | null): Chunk[] {
  const groups = new Map<string, Chunk[]>()
  for (const file of files) {
    const classId = classIdOf(file)
    const key = `${file.split}|${classId ?? ''}`
    const chunks = groups.get(key) ?? []
    let chunk = chunks[chunks.length - 1]
    const bytes = chunk?.files.reduce((n, f) => n + f.size, 0) ?? 0
    if (!chunk || chunk.files.length >= MAX_CHUNK_FILES || bytes + file.size > MAX_CHUNK_BYTES) {
      chunk = { split: file.split, classId, files: [] }
      chunks.push(chunk)
    }
    chunk.files.push(file)
    groups.set(key, chunks)
  }
  return [...groups.values()].flat()
}

export async function uploadStaged({
  projectId,
  files,
  pending,
  signal,
  onProgress,
  onSettled,
}: UploadRequest): Promise<UploadOutcome> {
  const outcome: UploadOutcome = { uploaded: 0, duplicates: 0, failed: 0, createdClasses: 0, cancelled: false }
  const total = files.length
  let settled = 0
  const settle = (uploadedIds: string[], errors: Record<string, string>) => {
    settled += uploadedIds.length + Object.keys(errors).length
    outcome.uploaded += uploadedIds.length
    outcome.failed += Object.keys(errors).length
    onSettled(uploadedIds, errors)
    onProgress({ settled, total })
  }
  onProgress({ settled, total })

  // 1. Classes for folders that don't exist yet. Re-list first: a previous attempt may have created some of
  //    them already, and creating them twice would fail.
  const classIds = new Map<string, string>()
  const wanted = new Map<string, PendingFolder>()
  for (const file of files) {
    if (isPendingKey(file.classKey)) {
      const folder = pending.get(file.classKey)
      if (folder) wanted.set(folder.key, folder)
    }
  }
  const failedClasses = new Map<string, string>()
  if (wanted.size > 0) {
    const existing = new Map(
      (await listClasses(projectId, undefined, signal)).classes.map((c) => [normalizeClassName(c.name), c.classId]),
    )
    for (const folder of wanted.values()) {
      const known = existing.get(normalizeClassName(folder.name))
      if (known) {
        classIds.set(folder.key, known)
        continue
      }
      try {
        classIds.set(folder.key, (await createClass(projectId, { name: folder.name }, undefined, signal)).class.classId)
        outcome.createdClasses++
      } catch (error) {
        if (signal.aborted) break
        failedClasses.set(folder.key, describeError(error, `Could not create class "${folder.name}"`))
      }
    }
  }

  const classIdOf = (file: StagedFile): string | null =>
    isPendingKey(file.classKey) ? (classIds.get(file.classKey) ?? null) : classIdOfKey(file.classKey)

  // Files whose new class could not be created are not sent unlabeled: they fail with the reason instead.
  const sendable: StagedFile[] = []
  const blocked: Record<string, string> = {}
  for (const file of files) {
    const reason = isPendingKey(file.classKey) ? failedClasses.get(file.classKey) : undefined
    if (reason) blocked[file.id] = reason
    else sendable.push(file)
  }
  if (Object.keys(blocked).length > 0) settle([], blocked)

  // 2. Chunked, concurrent upload.
  const queue = toChunks(sendable, classIdOf)
  const sendChunk = async (chunk: Chunk) => {
    try {
      const response = await uploadItems(
        projectId,
        { split: chunk.split, classId: chunk.classId ?? undefined, files: chunk.files.map((f) => f.file) },
        undefined,
        signal,
      )
      const uploadedIds: string[] = []
      const errors: Record<string, string> = {}
      chunk.files.forEach((file, i) => {
        const result = response.results[i]
        if (result?.status === 'fulfilled') {
          uploadedIds.push(file.id)
          if (result.value?.isDuplicate) outcome.duplicates++
        } else errors[file.id] = result?.reason ?? 'The server did not accept this file'
      })
      settle(uploadedIds, errors)
    } catch (error) {
      if (signal.aborted) return // cancelled mid-flight: leave the files staged, unmarked
      const reason = describeError(error, 'Upload request failed')
      settle([], Object.fromEntries(chunk.files.map((f) => [f.id, reason])))
    }
  }

  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      for (let chunk = queue.shift(); chunk && !signal.aborted; chunk = queue.shift()) await sendChunk(chunk)
    }),
  )
  outcome.cancelled = signal.aborted
  return outcome
}

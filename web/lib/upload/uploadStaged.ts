/**
 * The runner for file uploads (vision / audio / video): sends a job's files to the server.
 *
 * `POST /projects/:id/upload` takes one split + class per request, so files are grouped by that pair and each group
 * is cut into bounded chunks that go out a few at a time. Results are reported chunk by chunk: files the server took
 * are settled immediately, and files it refused are settled as failures with the reason, so a retry only resends
 * those. Classes for folders that don't exist yet are created first.
 *
 * The network calls come in through `FileApi`, so the chunking and the failure handling can be tested without a server.
 */

import { isAxiosError } from 'axios'
import { classIdOfKey, isPendingKey } from './fileTree'
import type { JobRunContext, JobUnit } from './jobs'
import { normalizeClassName } from './paths'
import type { PendingFolder } from './types'

export const MAX_CHUNK_FILES = 50
export const MAX_CHUNK_BYTES = 64 * 1024 * 1024
const CONCURRENCY = 3

export interface FilePayload {
  file: File
  /** `c:<classId>`, `n:<name>` for a class to create first, or `null`. */
  classKey: string | null
}

export type FileUnit = JobUnit<FilePayload>

export interface FileApi {
  listClasses: (projectId: string, signal: AbortSignal) => Promise<{ classId: string; name: string }[]>
  createClass: (projectId: string, name: string, signal: AbortSignal) => Promise<string>
  upload: (
    projectId: string,
    body: { split: string; classId?: string; files: File[] },
    signal: AbortSignal,
  ) => Promise<{ status: string; isDuplicate?: boolean; reason?: string | null }[]>
}

/** The server's `detail` when it sent one, else the transport error. */
export function describeError(error: unknown, fallback: string): string {
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (error.response) return `${fallback} (HTTP ${error.response.status})`
    return error.message
  }
  return error instanceof Error ? error.message : fallback
}

interface Chunk {
  split: FileUnit['split']
  classId: string | null
  units: FileUnit[]
}

export function toChunks(units: FileUnit[], classIdOf: (unit: FileUnit) => string | null): Chunk[] {
  const groups = new Map<string, { chunk: Chunk; bytes: number }[]>()
  for (const unit of units) {
    const classId = classIdOf(unit)
    const key = `${unit.split}|${classId ?? ''}`
    const chunks = groups.get(key) ?? []
    let last = chunks[chunks.length - 1]
    if (!last || last.chunk.units.length >= MAX_CHUNK_FILES || last.bytes + unit.weight > MAX_CHUNK_BYTES) {
      last = { chunk: { split: unit.split, classId, units: [] }, bytes: 0 }
      chunks.push(last)
    }
    last.chunk.units.push(unit)
    last.bytes += unit.weight
    groups.set(key, chunks)
  }
  return [...groups.values()].flatMap((chunks) => chunks.map((c) => c.chunk))
}

export function fileRunner(options: {
  api: FileApi
  projectId: string
  /** The folders waiting to become classes, as they were when the batch was committed. */
  pending: Map<string, PendingFolder>
}): (units: FileUnit[], ctx: JobRunContext<FilePayload>) => Promise<void> {
  const { api, projectId, pending } = options

  return async (units, ctx) => {
    const { signal } = ctx

    // 1. Classes for folders that don't exist yet. Re-list first: a previous attempt may have created some of
    //    them already, and creating them twice would fail.
    const classIds = new Map<string, string>()
    const wanted = new Map<string, PendingFolder>()
    for (const unit of units) {
      const key = unit.payload.classKey
      if (isPendingKey(key)) {
        const folder = pending.get(key)
        if (folder) wanted.set(folder.key, folder)
      }
    }
    const failedClasses = new Map<string, string>()
    if (wanted.size > 0) {
      ctx.phase('Creating classes')
      const existing = new Map(
        (await api.listClasses(projectId, signal)).map((c) => [normalizeClassName(c.name), c.classId]),
      )
      for (const folder of wanted.values()) {
        const known = existing.get(normalizeClassName(folder.name))
        if (known) {
          classIds.set(folder.key, known)
          continue
        }
        try {
          classIds.set(folder.key, await api.createClass(projectId, folder.name, signal))
          ctx.settle({ createdClasses: 1 })
        } catch (error) {
          if (signal.aborted) return
          failedClasses.set(folder.key, describeError(error, `Could not create class "${folder.name}"`))
        }
      }
      ctx.phase(null)
    }

    const classIdOf = (unit: FileUnit): string | null => {
      const key = unit.payload.classKey
      return isPendingKey(key) ? (classIds.get(key) ?? null) : classIdOfKey(key)
    }

    // Files whose new class could not be created are not sent unlabelled: they fail with the reason instead.
    const sendable: FileUnit[] = []
    const blocked: { unit: FileUnit; reason: string }[] = []
    for (const unit of units) {
      const key = unit.payload.classKey
      const reason = isPendingKey(key) ? failedClasses.get(key) : undefined
      if (reason) blocked.push({ unit, reason })
      else sendable.push(unit)
    }
    if (blocked.length > 0) ctx.settle({ failed: blocked })

    // 2. Chunked, concurrent upload.
    const queue = toChunks(sendable, classIdOf)
    const sendChunk = async (chunk: Chunk) => {
      try {
        const results = await api.upload(
          projectId,
          { split: chunk.split, classId: chunk.classId ?? undefined, files: chunk.units.map((u) => u.payload.file) },
          signal,
        )
        const ok: FileUnit[] = []
        const failed: { unit: FileUnit; reason: string }[] = []
        let duplicates = 0
        chunk.units.forEach((unit, i) => {
          const result = results[i]
          if (result?.status === 'fulfilled') {
            ok.push(unit)
            if (result.isDuplicate) duplicates++
          } else failed.push({ unit, reason: result?.reason ?? 'The server did not accept this file' })
        })
        ctx.settle({ ok, failed, duplicates })
      } catch (error) {
        if (signal.aborted) return // cancelled mid-flight: leave the files unsettled, so they stay retryable
        const reason = describeError(error, 'Upload request failed')
        ctx.settle({ failed: chunk.units.map((unit) => ({ unit, reason })) })
      }
    }

    await Promise.all(
      Array.from({ length: CONCURRENCY }, async () => {
        for (let chunk = queue.shift(); chunk && !signal.aborted; chunk = queue.shift()) await sendChunk(chunk)
      }),
    )
  }
}

/**
 * Bring-your-own models on the web: display constants, and one adapter per audience so the same components
 * serve "My models" (the user's own, /api/models) and the admin catalog (everyone's, /api/admin/models).
 */

import {
  adminCreateModel,
  adminDeleteModel,
  adminFinalizeModelUpload,
  adminListModels,
  adminModelUploadUrl,
  adminRetryModelValidation,
  adminUpdateModel,
} from './api/generated/admin/admin'
import type {
  CreateCustomModelBody,
  CustomModelOut,
  DeleteCustomModelResponse,
  UpdateCustomModelBody,
  UploadUrlBody,
  UploadUrlResponse,
} from './api/generated/models'
import {
  createMyModel,
  deleteMyModel,
  finalizeModelUpload,
  listMyModels,
  requestModelUploadUrl,
  retryModelValidation,
  updateMyModel,
} from './api/generated/models/models'
import { taskRegistry } from './tasks'

export const MODEL_STATUS_COLORS: Record<string, string> = {
  pending_upload: 'gray',
  uploaded: 'blue',
  validating: 'blue',
  ready: 'teal',
  failed: 'red',
}

export const MODEL_STATUS_LABELS: Record<string, string> = {
  pending_upload: 'Waiting for upload',
  uploaded: 'Queued',
  validating: 'Validating',
  ready: 'Ready',
  failed: 'Failed',
}

/** Statuses that change by themselves, so the list keeps refreshing while any model is in one. */
export const isValidating = (m: Pick<CustomModelOut, 'status'>) => m.status === 'uploaded' || m.status === 'validating'

/** A model's own `tasks` are ids; show what a person would call them. */
export const taskLabel = (id: string): string => (taskRegistry as Record<string, { label?: string }>)[id]?.label ?? id

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i++
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`
}

export interface ModelFilters {
  scope?: 'global' | 'private'
  status?: string
  q?: string
}

export interface ModelsApi {
  /** Which list this is, for cache keys, and whether it shows owners and can create global models. */
  audience: 'user' | 'admin'
  list: (filters: ModelFilters) => Promise<CustomModelOut[]>
  create: (body: CreateCustomModelBody) => Promise<CustomModelOut>
  update: (id: string, body: UpdateCustomModelBody) => Promise<CustomModelOut>
  remove: (id: string) => Promise<DeleteCustomModelResponse>
  uploadUrl: (id: string, body: UploadUrlBody) => Promise<UploadUrlResponse>
  finalize: (id: string) => Promise<CustomModelOut>
  retry: (id: string) => Promise<CustomModelOut>
}

/** Every custom-model query starts with this, so one invalidation refreshes them all. */
export const CUSTOM_MODELS_KEY = ['custom-models'] as const

export const userModelsApi: ModelsApi = {
  audience: 'user',
  list: async () => (await listMyModels()).models,
  create: async (body) => (await createMyModel(body)).model,
  update: async (id, body) => (await updateMyModel(id, body)).model,
  remove: (id) => deleteMyModel(id),
  uploadUrl: (id, body) => requestModelUploadUrl(id, body),
  finalize: async (id) => (await finalizeModelUpload(id)).model,
  retry: async (id) => (await retryModelValidation(id)).model,
}

export const adminModelsApi: ModelsApi = {
  audience: 'admin',
  list: async (filters) =>
    (await adminListModels({ scope: filters.scope, status: filters.status, q: filters.q || undefined })).models,
  create: async (body) => (await adminCreateModel(body)).model,
  update: async (id, body) => (await adminUpdateModel(id, body)).model,
  remove: (id) => adminDeleteModel(id),
  uploadUrl: (id, body) => adminModelUploadUrl(id, body),
  finalize: async (id) => (await adminFinalizeModelUpload(id)).model,
  retry: async (id) => (await adminRetryModelValidation(id)).model,
}

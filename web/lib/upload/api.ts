/** Binds the upload runners to the generated API client. Kept apart so the runners themselves stay testable. */

import { createClass, listClasses } from '@public/lib/api/generated/classes/classes'
import { createItems, uploadItems } from '@public/lib/api/generated/datasets/datasets'
import type { InlineApi } from './uploadInline'
import type { FileApi } from './uploadStaged'

export const fileApi: FileApi = {
  listClasses: async (projectId, signal) => (await listClasses(projectId, undefined, signal)).classes,
  createClass: async (projectId, name, signal) =>
    (await createClass(projectId, { name }, undefined, signal)).class.classId,
  upload: async (projectId, body, signal) => {
    const response = await uploadItems(projectId, body, undefined, signal)
    return response.results.map((r) => ({
      status: r.status,
      isDuplicate: r.value?.isDuplicate,
      reason: r.reason,
    }))
  },
}

export const inlineApi: InlineApi = {
  create: async (projectId, items, signal) => {
    const { failed } = await createItems(projectId, { items }, undefined, signal)
    return { failed }
  },
}

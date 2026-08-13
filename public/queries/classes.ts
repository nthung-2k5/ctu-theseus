import { createQueryKeys } from '@lukemorales/query-key-factory'
import { wrapEdenFn } from '@public/lib/eden-query'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export const classes = createQueryKeys('classes', {
  list: (projectId: string) => ({
    queryKey: [projectId],
    queryFn: wrapEdenFn(() => api.projects({ projectId }).classes.get()),
  }),
})

/**
 * Fetch all label classes for a project's dataset.
 */
export function useLabelClasses(projectId: string | undefined) {
  return useQuery({
    ...classes.list(projectId ?? ''),
    enabled: !!projectId,
  })
}

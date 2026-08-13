import { wrapEdenFn } from '@public/lib/eden-query'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

/**
 * Fetch paginated pool items for a project (draft by default, or a given
 * version/split).
 */
export function useProjectItems(
  projectId: string | undefined,
  query?: { versionId?: string; split?: 'train' | 'validation' | 'test'; page?: number; perPage?: number },
) {
  return useQuery({
    queryKey: ['items', projectId, query],
    queryFn: wrapEdenFn(() => api.projects({ projectId: projectId ?? '' }).items.get({ query })),
    enabled: !!projectId,
  })
}

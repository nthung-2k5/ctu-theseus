import { createQueryKeys } from '@lukemorales/query-key-factory'
import { useEdenQuery, wrapEdenFn as wrapEdenQueryFn } from '@public/lib/eden-query'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export const projects = createQueryKeys('projects', {
  all: null,
  detail: (projectId: string) => ({
    queryKey: [projectId],
    contextQueries: {
      summary: {
        queryKey: ['summary'],
        queryFn: wrapEdenQueryFn(api.projects({ projectId }).get),
      },
    },
  }),
})

export const useProjects = () => {
  return useEdenQuery({ ...projects.all, queryFn: api.projects.get })
}

/**
 * Fetch the full project detail (includes 1:1 dataset with draft/versions inline).
 */
export const useProjectDetail = (projectId: string) => {
  return useQuery(projects.detail(projectId)._ctx.summary)
}

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
      classes: {
        queryKey: ['classes'],
        queryFn: wrapEdenQueryFn(api.projects({ projectId }).classes.get),
      },
      dataset: {
        queryKey: ['dataset'],
        queryFn: wrapEdenQueryFn(api.projects({ projectId }).images.get),
        contextQueries: {
          stats: {
            queryKey: ['stat'],
            queryFn: wrapEdenQueryFn(api.projects({ projectId }).images.stats.get),
          },
          search: (query: DatasetSearchQuery) => ({
            queryKey: [query],
            queryFn: wrapEdenQueryFn(() => {
              console.log(query)
              return api.projects({ projectId }).images.get({ query })
            }),
          }),
        },
      },
    },
  }),
})

export type DatasetSearchQuery = NonNullable<Parameters<ReturnType<typeof api.projects>['images']['get']>[0]>['query']

export const useProjects = () => {
  return useEdenQuery({ ...projects.all, queryFn: api.projects.get })
}

export const useProjectDetail = (projectId: string) => {
  return useQuery(projects.detail(projectId)._ctx.summary)
}

export const useProjectClasses = (projectId: string) => {
  return useQuery(projects.detail(projectId)._ctx.classes)
}

export const useProjectDataset = (projectId: string, query: DatasetSearchQuery) => {
  return useQuery(projects.detail(projectId)._ctx.dataset._ctx.search(query))
}

export const useProjectDatasetStats = (projectId: string) => {
  return useQuery(projects.detail(projectId)._ctx.dataset._ctx.stats)
}

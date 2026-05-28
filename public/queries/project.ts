import { createQueryKeys } from '@lukemorales/query-key-factory'
import { useEdenQuery, wrapEdenFn as wrapEdenQueryFn } from '@public/lib/eden-query'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export const projects = createQueryKeys('projects', {
  all: null,
  detail: (projectId: string) => ({
    queryKey: [projectId],
    queryFn: wrapEdenQueryFn(api.projects({ projectId }).get),
  }),
  classes: (projectId: string) => ({
    queryKey: [projectId, 'classes'],
    queryFn: wrapEdenQueryFn(api.projects({ projectId }).classes.get),
  }),
})

export const useProjects = () => {
  return useEdenQuery({ ...projects.all, queryFn: api.projects.get })
}

export const useProjectDetail = (projectId: string) => {
  return useQuery(projects.detail(projectId))
}

export const useProjectClasses = (projectId: string) => {
  return useQuery(projects.classes(projectId))
}

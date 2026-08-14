import { rest, useEden } from '@public/lib/api'
import { useMutation, useQueryClient } from '@tanstack/react-query'

interface AssignVars {
  itemId: string
  /** The item's existing classification annotation id, if any — PATCHes it instead of creating a second one. */
  existingAnnotationId?: string
  classId?: string
  labelStructured?: unknown
}

/**
 * Creates or updates an item's classification-type annotation (see
 * server/lib/tasks/registry.ts — every stable task's ground truth is stored
 * as annotationType: 'classification', whether that means a class pick
 * (classId) or a regression target (labelStructured: {value})).
 *
 * itemId varies per call, so this is one reusable mutation rather than a
 * hook per item — matches the pattern already used for item/run deletion
 * elsewhere.
 */
export function useAssignAnnotation(projectId: string) {
  const eden = useEden()
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ itemId, existingAnnotationId, classId, labelStructured }: AssignVars) => {
      if (existingAnnotationId) {
        const { data, error } = await rest
          .annotations({ annotationId: existingAnnotationId })
          .patch({ classId, labelStructured })
        if (error) throw error
        return data
      }
      const { data, error } = await rest.items({ itemId }).annotations.post({
        annotationType: 'classification',
        classId,
        labelStructured,
      })
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).items.get.queryKey() })
    },
  })
}

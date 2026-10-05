import { notifications } from '@mantine/notifications'
import type { SPLIT_TYPES } from '@public/components/dataset/VersionBrowsing'
import {
  classifyItems,
  createAnnotation,
  deleteItems,
  setItemsSplit,
  updateAnnotation,
} from '@public/lib/api/generated/datasets/datasets'
import { invalidateProjectScope } from '@public/lib/queries'
import { useMutation, useQueryClient } from '@tanstack/react-query'

/**
 * The draft's item mutations, each notifying and invalidating project scope on
 * success (splits and classes both change counts rendered from project detail,
 * so invalidating just the items query isn't enough). Clearing the selection is
 * left to the caller: pass `onSuccess` to `mutate`.
 */
export function useItemMutations(projectId: string) {
  const queryClient = useQueryClient()
  const invalidate = () => invalidateProjectScope(queryClient, projectId)

  const onError = (message: string) => () => notifications.show({ title: 'Error', message, color: 'red' })

  const reassignSplit = useMutation({
    mutationFn: ({ itemIds, split }: { itemIds: string[]; split: (typeof SPLIT_TYPES)[number] }) =>
      setItemsSplit(projectId, { itemIds, split }),
    onSuccess: (_data, { itemIds }) => {
      notifications.show({ title: 'Split updated', message: `${itemIds.length} item(s) reassigned`, color: 'green' })
      invalidate()
    },
    onError: onError('Failed to reassign split'),
  })

  const assignClass = useMutation({
    mutationFn: ({ itemIds, classId }: { itemIds: string[]; classId: string }) =>
      classifyItems(projectId, { itemIds, classId }),
    onSuccess: (_data, { itemIds }) => {
      notifications.show({ title: 'Class assigned', message: `${itemIds.length} item(s) updated`, color: 'green' })
      invalidate()
    },
    onError: onError('Failed to assign class'),
  })

  const removeItems = useMutation({
    mutationFn: ({ itemIds }: { itemIds: string[] }) => deleteItems(projectId, { itemIds }),
    onSuccess: (_data, { itemIds }) => {
      notifications.show({
        title: 'Deleted',
        message: `${itemIds.length} item(s) removed from the draft`,
        color: 'green',
      })
      invalidate()
    },
    onError: onError('Bulk delete failed'),
  })

  // Free-text ground truth (captioning/ASR) has no shared value to bulk-assign
  // the way a class does, so it's edited per-item from the detail modal.
  const saveCaption = useMutation({
    mutationFn: async ({
      itemId,
      existingAnnotationId,
      text,
    }: {
      itemId: string
      existingAnnotationId: string | null
      text: string
    }) => {
      if (existingAnnotationId) await updateAnnotation(existingAnnotationId, { labelTextSequence: text })
      else await createAnnotation(itemId, { annotationType: 'text_sequence', labelTextSequence: text })
    },
    onSuccess: invalidate,
    onError: onError('Failed to save'),
  })

  return { reassignSplit, assignClass, removeItems, saveCaption }
}

import { notifications } from '@mantine/notifications'
import type { SPLIT_TYPES } from '@public/components/dataset/VersionBrowsing'
import {
  classifyItems,
  createAnnotation,
  deleteAnnotation,
  deleteItems,
  setItemsSplit,
  updateAnnotation,
} from '@public/lib/api/generated/datasets/datasets'
import type { Box } from '@public/lib/boxes'
import type { SelectionRequest } from '@public/lib/itemSelection'
import { invalidateProjectScope } from '@public/lib/queries'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { AxiosError } from 'axios'

/** The one-item case of a bulk selection. */
export const oneItem = (itemId: string): SelectionRequest => ({ itemIds: [itemId] })

/** The server's own explanation when it gave one (e.g. "More than 5000 items match"), else `fallback`. */
const failureMessage = (error: unknown, fallback: string) =>
  (error as AxiosError<{ error?: { message?: string } }> | undefined)?.response?.data?.error?.message ?? fallback

/**
 * The draft's item mutations, each notifying and invalidating project scope on
 * success (splits and classes both change counts rendered from project detail,
 * so invalidating just the items query isn't enough). Clearing the selection is
 * left to the caller: pass `onSuccess` to `mutate`. A selection is either explicit ids or a filter
 * the server resolves, so the counts in the notifications come from the response, not the request.
 */
export function useItemMutations(projectId: string) {
  const queryClient = useQueryClient()
  const invalidate = () => invalidateProjectScope(queryClient, projectId)

  const onError = (message: string) => (error: unknown) =>
    notifications.show({ title: 'Error', message: failureMessage(error, message), color: 'red' })

  const reassignSplit = useMutation({
    mutationFn: ({ selection, split }: { selection: SelectionRequest; split: (typeof SPLIT_TYPES)[number] }) =>
      setItemsSplit(projectId, { ...selection, split }),
    onSuccess: ({ updated }) => {
      notifications.show({ title: 'Split updated', message: `${updated.length} item(s) reassigned`, color: 'green' })
      invalidate()
    },
    onError: onError('Failed to reassign split'),
  })

  const assignClass = useMutation({
    mutationFn: ({ selection, classId }: { selection: SelectionRequest; classId: string }) =>
      classifyItems(projectId, { ...selection, classId }),
    onSuccess: ({ updated, failed }) => {
      notifications.show({
        title: 'Class assigned',
        message: `${updated} item(s) updated${failed ? `, ${failed} failed` : ''}`,
        color: failed ? 'yellow' : 'green',
      })
      invalidate()
    },
    onError: onError('Failed to assign class'),
  })

  const removeItems = useMutation({
    mutationFn: ({ selection }: { selection: SelectionRequest }) => deleteItems(projectId, selection),
    onSuccess: ({ results }) => {
      notifications.show({
        title: 'Deleted',
        message: `${results.length} item(s) removed from the draft`,
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

  // A regression target is a classification-typed annotation with no class; the number is `labelStructured.value`.
  const saveTarget = useMutation({
    mutationFn: async ({
      itemId,
      existingAnnotationId,
      value,
    }: {
      itemId: string
      existingAnnotationId: string | null
      value: number
    }) => {
      if (existingAnnotationId) await updateAnnotation(existingAnnotationId, { labelStructured: { value } })
      else await createAnnotation(itemId, { annotationType: 'classification', labelStructured: { value } })
    },
    onSuccess: invalidate,
    onError: onError('Failed to save'),
  })

  // Boxes: one annotation each, drawn on the image, so every change is a single-annotation call (no bulk class).
  const addBox = useMutation({
    mutationFn: ({ itemId, classId, box }: { itemId: string; classId: string; box: Box }) =>
      createAnnotation(itemId, { annotationType: 'bounding_box', classId, labelStructured: box }),
    onSuccess: invalidate,
    onError: onError('Failed to add the box'),
  })

  const updateBox = useMutation({
    mutationFn: ({ annotationId, classId, box }: { annotationId: string; classId?: string; box?: Box }) =>
      updateAnnotation(annotationId, {
        ...(classId !== undefined && { classId }),
        ...(box !== undefined && { labelStructured: box }),
      }),
    onSuccess: invalidate,
    onError: onError('Failed to change the box'),
  })

  const removeBox = useMutation({
    mutationFn: ({ annotationId }: { annotationId: string }) => deleteAnnotation(annotationId),
    onSuccess: invalidate,
    onError: onError('Failed to delete the box'),
  })

  // An image labelled with nothing in it: one class-less marker annotation, which a detector trains on as a
  // background image (as opposed to an image nobody has labelled, which it ignores).
  const markEmpty = useMutation({
    mutationFn: ({ itemId }: { itemId: string }) =>
      createAnnotation(itemId, { annotationType: 'bounding_box', labelStructured: { background: true } }),
    onSuccess: invalidate,
    onError: onError('Failed to mark the image'),
  })

  return { reassignSplit, assignClass, removeItems, saveCaption, saveTarget, addBox, updateBox, removeBox, markEmpty }
}

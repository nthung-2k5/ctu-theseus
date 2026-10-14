/**
 * Pressing Upload: decide the splits, move what is staged into an upload job, and let the job carry on in the
 * background. The staging area is emptied as soon as the job owns the files, so the user can stage the next batch
 * (or close the drawer) while this one is still going.
 */

import { Badge, Group, Stack, Text } from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { invalidateProjectScope } from '@public/lib/queries'
import { fileApi, inlineApi } from '@public/lib/upload/api'
import { type CommitSettings, newClassNamesOf, planFileCommit, planItemCommit } from '@public/lib/upload/commit'
import { startUploadJob, type UploadJobView } from '@public/lib/upload/jobs'
import type { SplitPlan } from '@public/lib/upload/splitPlan'
import { inlineRunner } from '@public/lib/upload/uploadInline'
import { fileRunner } from '@public/lib/upload/uploadStaged'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import type { StagedItem } from '@public/lib/uploadQueue'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useState } from 'react'

export const plural = (n: number, word: string) =>
  `${n.toLocaleString()} ${n === 1 ? word : `${word}${word.endsWith('s') ? 'es' : 's'}`}`

/** The one-line outcome of a run, for the notification that fires when a job ends. */
export function finishedToast(job: UploadJobView): void {
  const extras = [
    job.createdClasses > 0 && `${plural(job.createdClasses, 'class')} created`,
    job.duplicates > 0 && `${job.duplicates.toLocaleString()} already in the dataset`,
  ].filter(Boolean)
  const suffix = extras.length > 0 ? ` (${extras.join(', ')})` : ''
  if (job.status === 'cancelled') {
    notifications.show({
      title: 'Upload cancelled',
      message: `${job.uploaded.toLocaleString()} added before it stopped${suffix}`,
      color: 'yellow',
    })
  } else if (job.failed > 0) {
    notifications.show({
      title: 'Upload partially complete',
      message: `${job.uploaded.toLocaleString()} added, ${job.failed.toLocaleString()} failed${suffix}`,
      color: 'yellow',
    })
  } else {
    notifications.show({
      title: 'Upload complete',
      message: `${job.uploaded.toLocaleString()} added to the dataset${suffix}`,
      color: 'green',
    })
  }
}

/** Asks before folders become classes. Resolves `true` only when the user confirms. */
function confirmNewClasses(names: string[]): Promise<boolean> {
  return new Promise((resolve) => {
    let answered = false
    const answer = (value: boolean) => {
      if (answered) return
      answered = true
      resolve(value)
    }
    modals.openConfirmModal({
      title: `Create ${plural(names.length, 'new class')}?`,
      children: (
        <Stack gap="xs">
          <Text size="sm">
            These folders don't match an existing class. Uploading creates a class for each one and labels its files
            with it.
          </Text>
          <Group gap={6}>
            {names.slice(0, 30).map((name) => (
              <Badge key={name} variant="light" color="teal">
                {name}
              </Badge>
            ))}
            {names.length > 30 && <Text size="xs">…and {names.length - 30} more</Text>}
          </Group>
          <Text size="xs" c="dimmed">
            To use an existing class instead, cancel and choose "Use an existing class…" from the class's menu.
          </Text>
        </Stack>
      ),
      labels: { confirm: 'Create and upload', cancel: 'Cancel' },
      onConfirm: () => answer(true),
      onCancel: () => answer(false),
      onClose: () => answer(false),
    })
  })
}

/** A class too small to reach every split is worth a word before the draw is final. */
function warnSmallClasses(drawn: SplitPlan | null, nameOf: (key: string) => string) {
  if (!drawn || drawn.smallGroups.length === 0) return
  const names = drawn.smallGroups.map(nameOf)
  const shown = names.slice(0, 5).join(', ')
  notifications.show({
    title: 'Some classes are too small to split',
    message: `${shown}${names.length > 5 ? ` and ${names.length - 5} more` : ''} ${
      names.length === 1 ? 'has' : 'have'
    } fewer than 3 files, so cannot appear in all three splits.`,
    color: 'yellow',
  })
}

interface CommonOptions {
  projectId: string
  taskUsesClasses: boolean
  /** Whether a file or row with no class still counts as labelled (false where the label is drawn or typed later). */
  labelledWithoutClass: boolean
  settings: CommitSettings
}

export function useFileCommit({
  staging,
  classNames,
  projectId,
  taskUsesClasses,
  labelledWithoutClass,
  settings,
}: CommonOptions & {
  staging: FileStaging
  /** Existing classes by id, to name the ones a warning is about. */
  classNames: Map<string, string>
}) {
  const queryClient = useQueryClient()
  const [starting, setStarting] = useState(false)

  const start = useCallback(async () => {
    const options = { settings, taskUsesClasses, labelledWithoutClass }
    setStarting(true)
    try {
      // Hashing has to be done first: a file found to be a duplicate is better left unsent than sent and discarded.
      await staging.whenChecked()
      const state = staging.getState()
      const plan = planFileCommit(state.files.values(), options)
      if (plan.consumedIds.length === 0) return

      const newClasses = newClassNamesOf(plan.units, state.pending)
      if (newClasses.length > 0 && !(await confirmNewClasses(newClasses))) return

      // Re-read: the user may have edited the tree while the dialog was open, and only a current plan may go.
      const latest = staging.getState()
      const settled = latest === state ? plan : planFileCommit(latest.files.values(), options)
      if (settled.consumedIds.length === 0) return

      const nameOf = (key: string) =>
        key.startsWith('c:') ? (classNames.get(key.slice(2)) ?? 'A class') : (latest.pending.get(key)?.name ?? key)
      warnSmallClasses(settled.drawn, nameOf)
      staging.remove(settled.consumedIds)

      if (settled.units.length === 0) {
        notifications.show({
          message: `${plural(settled.alreadyThere, 'file')} already in the dataset — nothing to send`,
          color: 'blue',
        })
        return
      }
      startUploadJob({
        projectId,
        label: plural(settled.units.length, 'file'),
        units: settled.units,
        alreadyThere: settled.alreadyThere,
        run: fileRunner({ api: fileApi, projectId, pending: new Map(latest.pending) }),
        onFinished: (job) => {
          void invalidateProjectScope(queryClient, projectId)
          finishedToast(job)
        },
      })
    } finally {
      setStarting(false)
    }
  }, [staging, classNames, projectId, queryClient, settings, taskUsesClasses, labelledWithoutClass])

  return { start, starting }
}

export function useItemCommit({
  items,
  clear,
  classNames,
  projectId,
  taskUsesClasses,
  labelledWithoutClass,
  settings,
}: CommonOptions & {
  items: StagedItem[]
  clear: () => void
  classNames: Map<string, string>
}) {
  const queryClient = useQueryClient()

  const start = useCallback(() => {
    if (items.length === 0) return
    const { units, drawn } = planItemCommit(items, { settings, taskUsesClasses, labelledWithoutClass })
    warnSmallClasses(drawn, (classId) => classNames.get(classId) ?? 'A class')
    clear()
    startUploadJob({
      projectId,
      label: plural(units.length, 'item'),
      units,
      run: inlineRunner({ api: inlineApi, projectId }),
      onFinished: (job) => {
        void invalidateProjectScope(queryClient, projectId)
        finishedToast(job)
      },
    })
  }, [items, clear, classNames, projectId, queryClient, settings, taskUsesClasses, labelledWithoutClass])

  return { start }
}

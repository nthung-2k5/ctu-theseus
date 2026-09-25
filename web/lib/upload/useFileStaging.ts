/**
 * React glue for the file staging pipeline: drop/pick → ingest (unpack archives) → validate → detect the
 * layout → stage, with progress and cancellation. The state itself lives in the pure `fileTree.ts` reducer.
 */

import { notifications } from '@mantine/notifications'
import type { SplitType } from '@public/store/types'
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { buildIndex, type ClassRef, initialState, type StageOptions, stageBatch, treeReducer } from './fileTree'
import { ingest, type SourceFile, sourcesFromDataTransfer, sourcesFromFiles } from './ingest'
import type { DropContext, LayoutKind, RawEntry } from './types'
import { findDuplicates, validateFiles } from './validate'

export interface StagingBusy {
  label: string
  /** 0..1 */
  fraction: number
}

const collator = new Intl.Collator(undefined, { numeric: true })
const byPath = (a: RawEntry, b: RawEntry) => collator.compare(a.path.join('/'), b.path.join('/'))

export function useFileStaging({
  accept,
  taskUsesClasses,
  classes,
}: {
  accept?: string[]
  taskUsesClasses: boolean
  classes: ClassRef[]
}) {
  const [state, dispatch] = useReducer(treeReducer, undefined, initialState)
  const [busy, setBusy] = useState<StagingBusy | null>(null)

  // Handlers below run long after the render that created them; read the latest state through a ref.
  const stateRef = useRef(state)
  const abortRef = useRef<AbortController | null>(null)
  useEffect(() => {
    stateRef.current = state
  })

  const index = useMemo(() => buildIndex(state, classes, taskUsesClasses), [state, classes, taskUsesClasses])

  // Once a class a pending folder was waiting for exists (e.g. after an upload created it), fold the folder in.
  useEffect(() => {
    if (stateRef.current.pending.size > 0) dispatch({ type: 'reconcile', classes })
  }, [classes])

  const stage = useCallback(
    async (sources: SourceFile[], drop: DropContext) => {
      if (sources.length === 0) return
      if (abortRef.current) {
        notifications.show({ message: 'Still processing the previous drop', color: 'yellow' })
        return
      }
      const controller = new AbortController()
      abortRef.current = controller
      const { signal } = controller
      try {
        setBusy({ label: 'Reading files', fraction: 0 })
        const ingested = await ingest(sources, {
          signal,
          onProgress: ({ label, fraction }) => setBusy({ label, fraction }),
        })
        const entries = ingested.entries.sort(byPath)

        setBusy({ label: 'Checking files', fraction: 0 })
        const issues = await validateFiles(
          entries.map((e) => e.file),
          accept,
          { signal, onProgress: (fraction) => setBusy({ label: 'Checking files', fraction }) },
        )

        const options: StageOptions = { drop, taskUsesClasses, classes }
        const staged = stageBatch(entries, issues, stateRef.current.nextBatchId, options)
        dispatch({ type: 'add', staged, junk: ingested.junk, skipped: ingested.skipped })

        if (entries.length === 0 && ingested.skipped.length === 0 && ingested.junk === 0) {
          notifications.show({ message: 'Nothing to add', color: 'yellow' })
        }

        // Duplicates are only a hint, so don't hold the UI for them.
        const sizes = new Set(staged.files.map((f) => f.size))
        const candidates = [
          ...[...stateRef.current.files.values()].filter((f) => sizes.has(f.size)),
          ...staged.files,
        ].map((f) => ({ id: f.id, file: f.file }))
        void findDuplicates(candidates)
          .then((duplicates) => {
            if (duplicates.size > 0) dispatch({ type: 'duplicates', duplicates: Object.fromEntries(duplicates) })
          })
          .catch(() => {})
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          notifications.show({
            title: 'Could not add files',
            message: error instanceof Error ? error.message : 'Unexpected error',
            color: 'red',
          })
        }
      } finally {
        setBusy(null)
        if (abortRef.current === controller) abortRef.current = null
      }
    },
    [accept, taskUsesClasses, classes],
  )

  const stageFiles = useCallback(
    (files: Iterable<File>, drop: DropContext = {}) => {
      return stage(sourcesFromFiles(files), drop)
    },
    [stage],
  )

  const stageDrop = useCallback(
    async (dt: DataTransfer, drop: DropContext = {}) => {
      // Walking dropped folders is async but must start inside the drop event; `stage` takes over from there.
      const { sources, failed } = await sourcesFromDataTransfer(dt)
      if (failed.length > 0) {
        notifications.show({
          message: `${failed.length} dropped item(s) could not be read`,
          color: 'yellow',
        })
      }
      await stage(sources, drop)
    },
    [stage],
  )

  const relayout = useCallback(
    (kind: LayoutKind) =>
      dispatch({
        type: 'relayout',
        kind,
        options: { drop: stateRef.current.lastBatch?.drop ?? {}, taskUsesClasses, classes },
      }),
    [taskUsesClasses, classes],
  )

  const actions = useMemo(
    () => ({
      move: (ids: string[], target: { classKey?: string | null; split?: SplitType }) =>
        dispatch({ type: 'move', ids, ...target }),
      remove: (ids: string[]) => dispatch({ type: 'remove', ids }),
      mapFolder: (from: string, to: string | null) => dispatch({ type: 'mapFolder', from, to }),
      clear: () => dispatch({ type: 'clear' }),
      clearServerErrors: () => dispatch({ type: 'clearServerErrors' }),
      serverErrors: (errors: Record<string, string>) => dispatch({ type: 'serverErrors', errors }),
    }),
    [],
  )

  const renameFolder = useCallback(
    (key: string, name: string) => dispatch({ type: 'renameFolder', key, name, classes }),
    [classes],
  )

  const cancel = useCallback(() => abortRef.current?.abort(), [])

  return { state, index, busy, stageFiles, stageDrop, relayout, renameFolder, cancel, ...actions }
}

export type FileStaging = ReturnType<typeof useFileStaging>

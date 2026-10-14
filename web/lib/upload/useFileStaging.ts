/**
 * React glue for the file staging pipeline: drop/pick → ingest (unpack archives) → validate → detect the
 * layout → stage, with progress and cancellation. The state itself lives in the pure `fileTree.ts` reducer.
 *
 * Staging is followed by a background preflight (`preflight.ts`): every file is hashed and the server is asked which
 * contents the dataset already has, so files that need not be sent are marked before the user presses Upload.
 */

import { notifications } from '@mantine/notifications'
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import {
  buildIndex,
  type ClassRef,
  initialState,
  type StageOptions,
  stageBatch,
  type TreeAction,
  treeReducer,
} from './fileTree'
import { ingest, type SourceFile, sourcesFromDataTransfer, sourcesFromFiles } from './ingest'
import { canHash, hashFiles, planSkips } from './preflight'
import { findExisting } from './preflightRequest'
import type { DropContext, LayoutKind, RawEntry, StagedSplit } from './types'
import { validateFiles } from './validate'

export interface StagingBusy {
  label: string
  /** 0..1 */
  fraction: number
}

const collator = new Intl.Collator(undefined, { numeric: true })
const byPath = (a: RawEntry, b: RawEntry) => collator.compare(a.path.join('/'), b.path.join('/'))

export function useFileStaging({
  projectId,
  accept,
  taskUsesClasses,
  classes,
}: {
  projectId: string
  accept?: string[]
  taskUsesClasses: boolean
  classes: ClassRef[]
}) {
  const [state, rawDispatch] = useReducer(treeReducer, undefined, initialState)
  const [busy, setBusy] = useState<StagingBusy | null>(null)
  /** The background duplicate check. Unlike `busy` it never locks the tree: files can be edited while it runs. */
  const [checking, setChecking] = useState<StagingBusy | null>(null)

  // Handlers below run long after the render that created them, and the preflight must see a drop the moment it is
  // dispatched, so the latest state lives in a ref that every dispatch updates first. The reducer is pure, so
  // running it here and again inside React gives the same state.
  const stateRef = useRef(state)
  const dispatch = useCallback((action: TreeAction) => {
    stateRef.current = treeReducer(stateRef.current, action)
    rawDispatch(action)
  }, [])
  const abortRef = useRef<AbortController | null>(null)

  // Contents the dataset is known to hold, and the digests already asked about (each is asked once).
  const existingRef = useRef(new Set<string>())
  const askedRef = useRef(new Set<string>())
  const preflightRef = useRef<{ controller: AbortController; done: Promise<void> } | null>(null)

  const applySkips = useCallback(() => {
    const reasons = planSkips(stateRef.current.files.values(), existingRef.current)
    dispatch({ type: 'skip', reasons })
  }, [dispatch])

  // A removed file may have been the original of a duplicate, which then has to go after all.
  useEffect(() => {
    if (state.files.size > 0) applySkips()
  }, [state.files, applySkips])

  const startPreflight = useCallback((): Promise<void> => {
    if (!canHash()) return Promise.resolve()
    const previous = preflightRef.current
    previous?.controller.abort()
    const controller = new AbortController()
    const { signal } = controller
    const done = (async () => {
      await previous?.done
      const todo = [...stateRef.current.files.values()].filter((f) => !f.hash)
      if (todo.length === 0) return
      const hashed: Record<string, string> = {}
      try {
        setChecking({ label: 'Checking for duplicates', fraction: 0 })
        await hashFiles(
          todo.map((f) => ({ id: f.id, file: f.file })),
          {
            signal,
            onHashed: (id, hash) => {
              hashed[id] = hash
            },
            onProgress: (fraction) => setChecking({ label: 'Checking for duplicates', fraction }),
          },
        )
      } catch {
        // Aborted by a newer drop or a clear: what was hashed so far is still kept below.
      } finally {
        if (Object.keys(hashed).length > 0) dispatch({ type: 'hashed', hashes: hashed })
      }
      if (signal.aborted) return
      const fresh = [...new Set(Object.values(hashed))].filter((h) => !askedRef.current.has(h))
      try {
        for (const hash of fresh) askedRef.current.add(hash)
        for (const hash of await findExisting(projectId, fresh, signal)) existingRef.current.add(hash)
      } catch {
        // Advisory: without the server's answer only the duplicates inside this batch are found. Leave the digests
        // unasked, so the next drop tries them again.
        for (const hash of fresh) askedRef.current.delete(hash)
        if (signal.aborted) return
      }
      applySkips()
    })().finally(() => {
      if (preflightRef.current?.controller === controller) {
        preflightRef.current = null
        setChecking(null)
      }
    })
    preflightRef.current = { controller, done }
    return done
  }, [projectId, applySkips, dispatch])

  /** Resolves once the duplicate check for everything staged so far has finished (at once when none is running). */
  const whenChecked = useCallback(() => preflightRef.current?.done ?? Promise.resolve(), [])

  useEffect(() => () => preflightRef.current?.controller.abort(), [])

  const index = useMemo(() => buildIndex(state, classes, taskUsesClasses), [state, classes, taskUsesClasses])

  // Once a class a pending folder was waiting for exists (e.g. after an upload created it), fold the folder in.
  useEffect(() => {
    if (stateRef.current.pending.size > 0) dispatch({ type: 'reconcile', classes })
  }, [classes, dispatch])

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
        void startPreflight()
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
    [accept, taskUsesClasses, classes, startPreflight, dispatch],
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
    [taskUsesClasses, classes, dispatch],
  )

  const actions = useMemo(
    () => ({
      move: (ids: string[], target: { classKey?: string | null; split?: StagedSplit }) =>
        dispatch({ type: 'move', ids, ...target }),
      remove: (ids: string[]) => dispatch({ type: 'remove', ids }),
      mapFolder: (from: string, to: string | null) => dispatch({ type: 'mapFolder', from, to }),
      clear: () => {
        preflightRef.current?.controller.abort()
        dispatch({ type: 'clear' })
      },
    }),
    [dispatch],
  )

  const renameFolder = useCallback(
    (key: string, name: string) => dispatch({ type: 'renameFolder', key, name, classes }),
    [classes, dispatch],
  )

  const cancel = useCallback(() => abortRef.current?.abort(), [])

  /** The staged files right now, even from a callback that outlived the render that made it. */
  const getState = useCallback(() => stateRef.current, [])

  return {
    state,
    index,
    busy,
    checking,
    whenChecked,
    getState,
    stageFiles,
    stageDrop,
    relayout,
    renameFolder,
    cancel,
    ...actions,
  }
}

export type FileStaging = ReturnType<typeof useFileStaging>

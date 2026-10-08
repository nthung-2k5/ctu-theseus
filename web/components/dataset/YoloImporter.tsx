import { Alert, Badge, Button, Checkbox, Group, Progress, Select, Stack, Text, ThemeIcon } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { CloudArrowUpIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { createClass, listClasses } from '@public/lib/api/generated/classes/classes'
import { uploadItems } from '@public/lib/api/generated/datasets/datasets'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import { invalidateProjectScope } from '@public/lib/queries'
import { extractArchive } from '@public/lib/upload/archive'
import { archiveFormatFromName, normalizeClassName } from '@public/lib/upload/paths'
import {
  chunkBySplit,
  type ImportPlan,
  type ImportSplit,
  isImageName,
  isLabelName,
  isNamesFileName,
  pairKey,
  parseLabelFile,
  parseNamesFile,
  planImport,
  uploadBoxes,
  type YoloEntry,
} from '@public/lib/yoloImport'
import { useQueryClient } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import { useRef, useState } from 'react'

const UPLOAD_CONCURRENCY = 3
/** Label files are small; reading them a few hundred at a time keeps the tab responsive without a round trip each. */
const READ_BATCH = 200

interface Prepared {
  fileName: string
  plan: ImportPlan
  classNames: string[]
  /** Where the class names came from, so the review can say so. */
  namesFrom: string | null
  skippedFiles: number
}

const messageOf = (error: unknown, fallback: string) => {
  if (isAxiosError(error)) {
    return (error.response?.data as { error?: { message?: string } } | undefined)?.error?.message ?? fallback
  }
  return error instanceof Error ? error.message : fallback
}

const isYaml = (entry: YoloEntry) => /\.ya?ml$/i.test(entry.path.at(-1) ?? '')

const readTexts = async (entries: YoloEntry[]): Promise<Map<string, string>> => {
  const texts = new Map<string, string>()
  for (let start = 0; start < entries.length; start += READ_BATCH) {
    const batch = entries.slice(start, start + READ_BATCH)
    const read = await Promise.all(batch.map((e) => e.file.text()))
    batch.forEach((entry, i) => {
      texts.set(pairKey(entry.path), read[i])
    })
  }
  return texts
}

/**
 * Imports a labelled YOLO dataset from one archive: images, their `.txt` label files and the class names
 * (`data.yaml` or `classes.txt`). Classes the project lacks are created, then the images go up in chunks together
 * with their boxes, through the same upload route as any image, so storage and de-duplication behave as usual.
 */
export const YoloImporter = ({ projectId, onDone }: { projectId: string; onDone?: () => void }) => {
  const queryClient = useQueryClient()
  const [phase, setPhase] = useState<'idle' | 'reading' | 'ready' | 'importing' | 'done'>('idle')
  const [readProgress, setReadProgress] = useState(0)
  const [prepared, setPrepared] = useState<Prepared | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [defaultSplit, setDefaultSplit] = useState<ImportSplit>('train')
  const [missingAsEmpty, setMissingAsEmpty] = useState(false)
  const [progress, setProgress] = useState({ settled: 0, total: 0 })
  const [outcome, setOutcome] = useState<{
    uploaded: number
    duplicates: number
    failed: string[]
    createdClasses: number
  } | null>(null)
  // The extracted dataset stays here (not in state: it is large and never rendered), so changing an option re-plans.
  const dataset = useRef<{ images: YoloEntry[]; classNames: string[] } | null>(null)
  const abort = useRef<AbortController | null>(null)

  // The label files' text, kept to re-plan when an option changes without reading the archive again.
  const labelTexts = useRef<Map<string, string>>(new Map())

  const plan = (images: YoloEntry[], labels: Map<string, string>, classCount: number, asEmpty: boolean) =>
    planImport(images, labels, classCount, { missingAsEmpty: asEmpty })

  const handleDrop = async (files: File[]) => {
    const file = files[0]
    if (!file) return
    const format = archiveFormatFromName(file.name)
    if (!format) {
      setError('That is not a .zip, .tar or .tar.gz archive.')
      return
    }
    setError(null)
    setPhase('reading')
    setReadProgress(0)
    abort.current = new AbortController()
    try {
      const { entries, skipped } = await extractArchive(file, format, {
        signal: abort.current.signal,
        onProgress: setReadProgress,
      })
      const all: YoloEntry[] = entries.map((e) => ({ path: e.path, file: e.file }))
      const images = all.filter((e) => isImageName(e.path[e.path.length - 1]))
      const labelEntries = all.filter((e) => isLabelName(e.path[e.path.length - 1]))
      if (images.length === 0) throw new Error('The archive has no .jpg or .png images.')

      // Prefer a data.yaml, then the shallowest names file: a dataset's own file, not one inside a sub-folder.
      const namesEntries = all
        .filter((e) => isNamesFileName(e.path[e.path.length - 1]))
        .sort((a, b) => Number(!isYaml(a)) - Number(!isYaml(b)) || a.path.length - b.path.length)
      let classNames: string[] | null = null
      let namesFrom: string | null = null
      for (const candidate of namesEntries) {
        classNames = parseNamesFile(candidate.path[candidate.path.length - 1], await candidate.file.text())
        if (classNames) {
          namesFrom = candidate.path.join('/')
          break
        }
      }

      const texts = await readTexts(labelEntries)
      if (!classNames) {
        // No names file: label files still say how many classes there are, so name them by their index.
        let highest = -1
        for (const text of texts.values()) {
          for (const { classIndex } of parseLabelFile(text, Number.MAX_SAFE_INTEGER).boxes) {
            highest = Math.max(highest, classIndex)
          }
        }
        if (highest < 0) throw new Error('No class names (data.yaml or classes.txt) and no boxes in the label files.')
        classNames = Array.from({ length: highest + 1 }, (_, i) => `class ${i}`)
      }

      labelTexts.current = texts
      dataset.current = { images, classNames }
      setPrepared({
        fileName: file.name,
        plan: plan(images, texts, classNames.length, missingAsEmpty),
        classNames,
        namesFrom,
        skippedFiles: skipped.length,
      })
      setPhase('ready')
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(messageOf(e, 'The archive could not be read.'))
      setPhase('idle')
    }
  }

  const changeMissingAsEmpty = (next: boolean) => {
    setMissingAsEmpty(next)
    if (!dataset.current || !prepared) return
    setPrepared({
      ...prepared,
      plan: plan(dataset.current.images, labelTexts.current, dataset.current.classNames.length, next),
    })
  }

  const reset = () => {
    abort.current?.abort()
    dataset.current = null
    labelTexts.current = new Map()
    setPrepared(null)
    setOutcome(null)
    setError(null)
    setPhase('idle')
  }

  const handleImport = async () => {
    if (!prepared) return
    setPhase('importing')
    setError(null)
    abort.current = new AbortController()
    const { signal } = abort.current
    const result = { uploaded: 0, duplicates: 0, failed: [] as string[], createdClasses: 0 }
    try {
      // 1. A class per name. Re-list first: an earlier, interrupted import may already have created some.
      const existing = new Map(
        (await listClasses(projectId, undefined, signal)).classes.map((c) => [normalizeClassName(c.name), c.classId]),
      )
      const classIds: string[] = []
      for (const name of prepared.classNames) {
        let id = existing.get(normalizeClassName(name))
        if (!id) {
          id = (await createClass(projectId, { name }, undefined, signal)).class.classId
          existing.set(normalizeClassName(name), id)
          result.createdClasses++
        }
        classIds.push(id)
      }

      // 2. The images, a few requests at a time, each with its boxes.
      const chunks = chunkBySplit(prepared.plan.items, defaultSplit)
      const total = prepared.plan.items.length
      let settled = 0
      let next = 0
      setProgress({ settled, total })
      const worker = async () => {
        while (next < chunks.length && !signal.aborted) {
          const chunk = chunks[next++]
          try {
            const response = await uploadItems(
              projectId,
              {
                files: chunk.items.map((i) => i.file),
                split: chunk.split,
                boxes: JSON.stringify(uploadBoxes(chunk.items, classIds)),
              },
              undefined,
              signal,
            )
            for (const r of response.results) {
              if (r.status === 'fulfilled') {
                result.uploaded++
                if (r.value?.isDuplicate) result.duplicates++
              } else {
                result.failed.push(r.reason ?? 'Rejected')
              }
            }
          } catch (e) {
            if (signal.aborted) return
            result.failed.push(...chunk.items.map(() => messageOf(e, 'The upload failed')))
          }
          settled += chunk.items.length
          setProgress({ settled, total })
        }
      }
      await Promise.all(Array.from({ length: UPLOAD_CONCURRENCY }, worker))
      setOutcome(result)
      setPhase('done')
      invalidateProjectScope(queryClient, projectId)
      onDone?.()
    } catch (e) {
      if (!signal.aborted) setError(messageOf(e, 'The import failed.'))
      setOutcome(result.uploaded > 0 ? result : null)
      invalidateProjectScope(queryClient, projectId)
      setPhase(result.uploaded > 0 ? 'done' : 'ready')
    }
  }

  if (phase === 'idle' || phase === 'reading') {
    return (
      <Stack gap="sm">
        {error && (
          <Alert color="red" icon={<WarningCircleIcon size={16} />} p="xs">
            {error}
          </Alert>
        )}
        <Dropzone
          onDrop={handleDrop}
          loading={phase === 'reading'}
          multiple={false}
          maxSize={2 * 1024 * 1024 * 1024}
          radius="md"
          p="xl"
          style={{ borderStyle: 'dashed', borderWidth: 2 }}
        >
          <Stack align="center" gap="xs">
            <ThemeIcon size={48} variant="light" radius="xl">
              <CloudArrowUpIcon size={26} />
            </ThemeIcon>
            <Text size="sm" fw={500}>
              Drop a YOLO dataset archive here, or click to browse
            </Text>
            <Text size="xs" c="dimmed" ta="center">
              A .zip or .tar(.gz) with the images, one .txt label file per image (images/… and labels/… folders, as
              Ultralytics lays them out) and the class names in data.yaml or classes.txt.
            </Text>
          </Stack>
        </Dropzone>
        {phase === 'reading' && (
          <Progress value={Math.round(readProgress * 100)} size="sm" animated aria-label="Reading the archive" />
        )}
      </Stack>
    )
  }

  if (!prepared) return null
  const { plan: current } = prepared
  const labelled = current.items.filter((i) => i.boxes !== null).length
  const boxCount = current.items.reduce((n, i) => n + (i.boxes?.length ?? 0), 0)
  const importing = phase === 'importing'

  return (
    <Stack gap="sm">
      {error && (
        <Alert color="red" icon={<WarningCircleIcon size={16} />} p="xs">
          {error}
        </Alert>
      )}

      <div>
        <Text size="sm" fw={600}>
          {prepared.fileName}
        </Text>
        <Text size="sm">
          {current.items.length} image{current.items.length === 1 ? '' : 's'} · {labelled} labelled · {boxCount} box
          {boxCount === 1 ? '' : 'es'}
        </Text>
      </div>

      <div>
        <Text size="xs" c="dimmed" mb={4}>
          {prepared.classNames.length} class{prepared.classNames.length === 1 ? '' : 'es'}
          {prepared.namesFrom ? ` from ${prepared.namesFrom}` : ' (no names file: named by index)'}. Ones the project
          does not have yet are created.
        </Text>
        <Group gap={4}>
          {prepared.classNames.slice(0, 30).map((name, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: class names are not unique keys, their index is
            <Badge key={i} variant="light" size="sm" tt="none">
              {name}
            </Badge>
          ))}
          {prepared.classNames.length > 30 && <Badge variant="outline">+{prepared.classNames.length - 30}</Badge>}
        </Group>
      </div>

      {(current.withoutLabels > 0 ||
        current.skippedLines > 0 ||
        current.orphanLabels > 0 ||
        prepared.skippedFiles > 0) && (
        <Alert color="yellow" p="xs" icon={<WarningCircleIcon size={16} />}>
          <Stack gap={2}>
            {current.withoutLabels > 0 && (
              <Text size="xs">
                {current.withoutLabels} image{current.withoutLabels === 1 ? ' has' : 's have'} no label file:{' '}
                {missingAsEmpty ? 'imported as having no objects.' : 'imported unlabelled.'}
              </Text>
            )}
            {current.skippedLines > 0 && (
              <Text size="xs">
                {current.skippedLines} label line{current.skippedLines === 1 ? '' : 's'} could not be read as a box
                (polygons, unknown classes, boxes outside the image) and were left out.
              </Text>
            )}
            {current.orphanLabels > 0 && (
              <Text size="xs">
                {current.orphanLabels} label file{current.orphanLabels === 1 ? ' has' : 's have'} no matching image.
              </Text>
            )}
            {prepared.skippedFiles > 0 && (
              <Text size="xs">{prepared.skippedFiles} archive entries were skipped as unreadable or too large.</Text>
            )}
          </Stack>
        </Alert>
      )}

      {phase !== 'done' && (
        <>
          <Select
            label="Split for images not in a train / val / test folder"
            size="xs"
            data={SPLIT_OPTIONS}
            value={defaultSplit}
            allowDeselect={false}
            disabled={importing}
            onChange={(v) => v && setDefaultSplit(v as ImportSplit)}
          />
          <Checkbox
            size="xs"
            label="Treat images without a label file as having no objects"
            description="Some datasets mark background images by leaving the label file out."
            checked={missingAsEmpty}
            disabled={importing}
            onChange={(e) => changeMissingAsEmpty(e.currentTarget.checked)}
          />
        </>
      )}

      {importing && (
        <Progress
          value={progress.total ? Math.round((progress.settled / progress.total) * 100) : 0}
          size="sm"
          animated
          aria-label="Uploading"
        />
      )}

      {phase === 'done' && outcome && (
        <Alert color={outcome.failed.length > 0 ? 'yellow' : 'teal'} p="xs">
          <Stack gap={2}>
            <Text size="sm" fw={600}>
              Imported {outcome.uploaded} image{outcome.uploaded === 1 ? '' : 's'}
              {outcome.duplicates > 0 ? ` (${outcome.duplicates} already in the project, left as they were)` : ''}
            </Text>
            {outcome.createdClasses > 0 && (
              <Text size="xs">
                Created {outcome.createdClasses} class{outcome.createdClasses === 1 ? '' : 'es'}.
              </Text>
            )}
            {outcome.failed.length > 0 && (
              <Text size="xs">
                {outcome.failed.length} failed. First reason: {outcome.failed[0]}
              </Text>
            )}
          </Stack>
        </Alert>
      )}

      <Group justify="flex-end">
        {phase === 'done' ? (
          <Button onClick={reset}>Import another</Button>
        ) : (
          <>
            <Button variant="default" onClick={reset} disabled={importing}>
              Choose a different archive
            </Button>
            <Button onClick={handleImport} loading={importing} disabled={current.items.length === 0}>
              Import {current.items.length} image{current.items.length === 1 ? '' : 's'}
            </Button>
          </>
        )}
      </Group>
    </Stack>
  )
}

/**
 * Batch importer for text tasks: the counterpart of typing one item into the text box.
 *
 * Accepts many files in one drop and turns them into staged items, so the split and class of each stay
 * editable in the preview pane until the user presses Upload:
 *  - `.txt` / `.md` — one item per file.
 *  - `.csv` / `.tsv` / `.jsonl` — one item per row; the user picks which column holds the text and,
 *    for tasks with label classes, which (optional) column names the class.
 *
 * Without a label column (or on a row whose label cell is empty) an item takes the class chosen in the panel
 * above, like typed text does. A label value with no matching class stages unassigned rather than blocking the
 * import — or, if the user ticks "create missing classes", becomes a new class first, which is how a project
 * with no classes yet can be filled from a labelled file in one go.
 */

import { Alert, Button, Checkbox, Group, Select, Stack, Table, Text } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { CloudArrowUpIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { createClass } from '@public/lib/api/generated/classes/classes'
import { invalidateProjectScope } from '@public/lib/queries'
import type { StagedSplit } from '@public/lib/upload/types'
import type { StagedDraft } from '@public/lib/uploadQueue'
import type { LabelClass } from '@public/store/types'
import { useQueryClient } from '@tanstack/react-query'
import Papa from 'papaparse'
import { useMemo, useState } from 'react'

const PLAIN_EXTENSIONS = ['txt', 'md']
const TABLE_EXTENSIONS = ['csv', 'tsv', 'jsonl']
const ACCEPT = [...PLAIN_EXTENSIONS, ...TABLE_EXTENSIONS].map((ext) => `.${ext}`)
/** A text file bigger than this is almost certainly not one training example. */
const MAX_TEXT_FILE_BYTES = 1024 * 1024
const PREVIEW_ROWS = 5
/** Mirrors the server's class-name limit. */
const MAX_CLASS_NAME_LENGTH = 100
/** More distinct values than this in the class column almost certainly means the wrong column was picked. */
const MAX_CLASSES_TO_CREATE = 100

interface ParsedTable {
  fileName: string
  headers: string[]
  rows: Record<string, string>[]
}

const extensionOf = (name: string) => name.slice(name.lastIndexOf('.') + 1).toLowerCase()

function toDraft(text: string, sourceName: string, split: StagedSplit, classId: string | null): StagedDraft {
  return {
    kind: 'text',
    name: text.length > 80 ? `${text.slice(0, 80)}…` : text,
    detail: `${text.length} characters`,
    sourceName,
    split,
    classId,
    text,
  }
}

async function parseTable(file: File): Promise<ParsedTable> {
  if (extensionOf(file.name) === 'jsonl') {
    const rows: Record<string, string>[] = []
    const headers = new Set<string>()
    for (const line of (await file.text()).split(/\r?\n/)) {
      if (!line.trim()) continue
      const value: unknown = JSON.parse(line)
      const record =
        value !== null && typeof value === 'object' && !Array.isArray(value)
          ? (value as Record<string, unknown>)
          : { text: value }
      const row: Record<string, string> = {}
      for (const [key, v] of Object.entries(record)) {
        headers.add(key)
        row[key] = typeof v === 'string' ? v : JSON.stringify(v)
      }
      rows.push(row)
    }
    return { fileName: file.name, headers: [...headers], rows }
  }

  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      delimiter: extensionOf(file.name) === 'tsv' ? '\t' : undefined,
      complete: (result) => resolve({ fileName: file.name, headers: result.meta.fields ?? [], rows: result.data }),
      error: reject,
    })
  })
}

export function TextBatchImporter({
  projectId,
  taskUsesClasses,
  split,
  classId,
  classes,
  onStage,
}: {
  projectId: string
  /** Whether the task labels items with classes at all — decides if the class column can be picked. */
  taskUsesClasses: boolean
  /** Defaults from the panel above; every staged item can still be retargeted in the preview. */
  split: StagedSplit
  classId: string | null
  classes: LabelClass[]
  onStage: (drafts: StagedDraft[]) => void
}) {
  // Tables wait here until their columns are picked; the head of the list is the one on screen.
  const [tables, setTables] = useState<ParsedTable[]>([])
  const [textColumn, setTextColumn] = useState<string | null>(null)
  const [labelColumn, setLabelColumn] = useState<string | null>(null)
  // `null` = not chosen yet: defaults on when the project has no classes, since nothing could match otherwise.
  const [createMissingChoice, setCreateMissingChoice] = useState<boolean | null>(null)
  const [staging, setStaging] = useState(false)
  const queryClient = useQueryClient()
  const table = tables[0]

  const classIdByName = useMemo(() => new Map(classes.map((c) => [c.name.trim().toLowerCase(), c.classId])), [classes])

  const unmatchedValues = useMemo(() => {
    if (!table || !labelColumn) return []
    const seen = new Set<string>()
    for (const row of table.rows) {
      const value = row[labelColumn]?.trim()
      if (value && !classIdByName.has(value.toLowerCase())) seen.add(value)
    }
    return [...seen]
  }, [table, labelColumn, classIdByName])

  const tooManyToCreate = unmatchedValues.length > MAX_CLASSES_TO_CREATE
  const createMissing =
    (createMissingChoice ?? classes.length === 0) &&
    unmatchedValues.length > 0 &&
    !tooManyToCreate &&
    unmatchedValues.every((v) => v.length <= MAX_CLASS_NAME_LENGTH)

  const handleDrop = async (files: File[]) => {
    const drafts: StagedDraft[] = []
    const rejected: string[] = []
    const parsedTables: ParsedTable[] = []

    for (const file of files) {
      const ext = extensionOf(file.name)
      try {
        if (PLAIN_EXTENSIONS.includes(ext)) {
          if (file.size > MAX_TEXT_FILE_BYTES) {
            rejected.push(`${file.name} (over ${MAX_TEXT_FILE_BYTES / 1024 / 1024} MB)`)
            continue
          }
          const text = (await file.text()).trim()
          if (text) drafts.push(toDraft(text, file.name, split, classId))
          else rejected.push(`${file.name} (empty)`)
        } else if (TABLE_EXTENSIONS.includes(ext)) {
          const parsed = await parseTable(file)
          if (parsed.headers.length === 0 || parsed.rows.length === 0) rejected.push(`${file.name} (no rows)`)
          else parsedTables.push(parsed)
        } else {
          rejected.push(`${file.name} (unsupported type)`)
        }
      } catch {
        rejected.push(`${file.name} (could not be read)`)
      }
    }

    if (drafts.length > 0) onStage(drafts)
    if (parsedTables.length > 0) {
      setTables((prev) => [...prev, ...parsedTables])
      if (tables.length === 0) {
        setTextColumn(null)
        setLabelColumn(null)
      }
    }
    if (rejected.length > 0) {
      notifications.show({
        title: `${rejected.length} file(s) skipped`,
        message: rejected.slice(0, 5).join(', ') + (rejected.length > 5 ? `, +${rejected.length - 5} more` : ''),
        color: 'yellow',
      })
    }
  }

  const nextTable = () => {
    setTables((prev) => prev.slice(1))
    setTextColumn(null)
    setLabelColumn(null)
    setCreateMissingChoice(null)
  }

  /** Creates the classes the label column names but the project lacks; returns the full name → id lookup. */
  const resolveClasses = async () => {
    const lookup = new Map(classIdByName)
    if (!createMissing) return lookup
    const failed: string[] = []
    for (const name of unmatchedValues) {
      try {
        const { class: created } = await createClass(projectId, { name })
        lookup.set(created.name.trim().toLowerCase(), created.classId)
      } catch {
        failed.push(name)
      }
    }
    await invalidateProjectScope(queryClient, projectId)
    if (failed.length > 0) {
      notifications.show({
        title: `${failed.length} class(es) not created`,
        message: `${failed.slice(0, 5).join(', ')} — those rows stage without a class.`,
        color: 'yellow',
      })
    }
    return lookup
  }

  const handleStage = async () => {
    if (!table || !textColumn) return
    setStaging(true)
    try {
      const lookup = await resolveClasses()
      const drafts: StagedDraft[] = []
      for (const row of table.rows) {
        const text = row[textColumn]?.trim()
        if (!text) continue
        const label = labelColumn ? row[labelColumn]?.trim() : ''
        // An empty label cell falls back to the default class rather than leaving the row unassigned.
        const rowClassId = label ? (lookup.get(label.toLowerCase()) ?? null) : classId
        drafts.push(toDraft(text, table.fileName, split, rowClassId))
      }
      if (drafts.length === 0) {
        notifications.show({ title: 'Nothing to add', message: `Column "${textColumn}" is empty`, color: 'yellow' })
        return
      }
      onStage(drafts)
      nextTable()
    } finally {
      setStaging(false)
    }
  }

  return (
    <Stack gap="md">
      {table ? (
        <>
          <Text size="sm" fw={600}>
            {table.fileName}
            {tables.length > 1 ? ` (+${tables.length - 1} more waiting)` : ''}
          </Text>
          <Group grow align="flex-start">
            <Select
              label="Text column"
              placeholder="Select the text column"
              data={table.headers}
              value={textColumn}
              onChange={setTextColumn}
            />
            {taskUsesClasses && (
              <Select
                label="Class column"
                description="Optional — values are matched to class names"
                placeholder="Use the class above"
                data={table.headers.filter((h) => h !== textColumn)}
                value={labelColumn}
                onChange={setLabelColumn}
                clearable
              />
            )}
          </Group>

          {unmatchedValues.length > 0 && (
            <Alert icon={<WarningCircleIcon size={16} />} color="yellow" title="Unmatched class names">
              {unmatchedValues.slice(0, 10).join(', ')}
              {unmatchedValues.length > 10 ? `, +${unmatchedValues.length - 10} more` : ''}
              {tooManyToCreate
                ? ` — ${unmatchedValues.length} distinct values is too many to be classes; check the column.`
                : ' — no class of that name exists yet.'}
              {!tooManyToCreate && (
                <Checkbox
                  mt="xs"
                  size="xs"
                  label={`Create ${unmatchedValues.length} missing class(es) when adding`}
                  description="Otherwise these rows stage without a class; assign one in the preview."
                  checked={createMissing}
                  onChange={(e) => setCreateMissingChoice(e.currentTarget.checked)}
                />
              )}
            </Alert>
          )}

          <Table.ScrollContainer minWidth={400}>
            <Table striped withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  {table.headers.map((h) => (
                    <Table.Th key={h}>{h}</Table.Th>
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {table.rows.slice(0, PREVIEW_ROWS).map((row, i) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: static preview slice, never reordered/filtered
                  <Table.Tr key={`preview-${i}`}>
                    {table.headers.map((h) => (
                      <Table.Td key={h} maw={240} style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {row[h]}
                      </Table.Td>
                    ))}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          <Text size="xs" c="dimmed">
            Showing {Math.min(PREVIEW_ROWS, table.rows.length)} of {table.rows.length} rows.
          </Text>

          <Group justify="flex-end">
            <Button variant="subtle" onClick={nextTable} disabled={staging}>
              {tables.length > 1 ? 'Skip this file' : 'Cancel'}
            </Button>
            <Button disabled={!textColumn} loading={staging} onClick={handleStage}>
              Add {table.rows.length} row(s)
            </Button>
          </Group>
        </>
      ) : (
        <Dropzone onDrop={handleDrop} accept={ACCEPT} radius="md">
          <Group justify="center" gap="md" py="xl" style={{ pointerEvents: 'none' }}>
            <CloudArrowUpIcon size={24} />
            <div>
              <Text size="sm" fw={600}>
                Drop text files here or click to browse
              </Text>
              <Text size="xs" c="dimmed">
                .txt / .md: one item per file. .csv / .tsv / .jsonl: one item per row.
              </Text>
            </div>
          </Group>
        </Dropzone>
      )}
    </Stack>
  )
}

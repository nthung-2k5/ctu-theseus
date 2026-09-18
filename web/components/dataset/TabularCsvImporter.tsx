/**
 * CSV importer for tabular tasks — the gap DataPage's old RecordEntryPanel
 * explicitly punted on: pasting raw JSON has no way to attach a target
 * label, so tabular_classification/regression snapshots built from it had a
 * null label column. This closes that loop: upload -> preview -> pick the
 * target column -> stage one item per row, with both featuresJson AND the
 * matching class resolved from the target value.
 *
 * Rows are handed to the Upload page's staging queue rather than created
 * directly, so the split and class of every row stay editable in the preview
 * pane until the user presses Upload. That also means a target value with no
 * matching class no longer blocks the import — those rows stage unassigned
 * and can be fixed in place.
 */

import { Alert, Button, Group, Select, Stack, Table, Text } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { CloudArrowUpIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import type { StagedDraft } from '@public/lib/uploadQueue'
import type { LabelClass, SplitType } from '@public/store/types'
import Papa from 'papaparse'
import { useMemo, useState } from 'react'

/** Best-effort scalar coercion for feature columns — numeric strings become numbers, everything else stays a string. */
function coerce(value: string): string | number {
  if (value.trim() === '') return value
  const num = Number(value)
  return Number.isFinite(num) ? num : value
}

export function TabularCsvImporter({
  requiresLabelClasses,
  classes,
  onStage,
}: {
  requiresLabelClasses: boolean
  classes: LabelClass[]
  onStage: (drafts: StagedDraft[]) => void
}) {
  const [headers, setHeaders] = useState<string[]>([])
  const [rows, setRows] = useState<Record<string, string>[]>([])
  const [targetColumn, setTargetColumn] = useState<string | null>(null)
  const [split, setSplit] = useState<SplitType>('train')
  const [fileName, setFileName] = useState<string>('')

  const handleDrop = (files: File[]) => {
    const file = files[0]
    if (!file) return
    setFileName(file.name)
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        setHeaders(result.meta.fields ?? [])
        setRows(result.data)
        setTargetColumn(null)
      },
      error: () => {
        notifications.show({ title: 'Error', message: 'Could not parse CSV file', color: 'red' })
      },
    })
  }

  const classIdByName = useMemo(() => new Map(classes.map((c) => [c.name.trim().toLowerCase(), c.classId])), [classes])

  const unmatchedValues = useMemo(() => {
    if (!requiresLabelClasses || !targetColumn) return []
    const seen = new Set<string>()
    for (const row of rows) {
      const value = row[targetColumn]?.trim()
      if (value && !classIdByName.has(value.toLowerCase())) seen.add(value)
    }
    return [...seen]
  }, [requiresLabelClasses, targetColumn, rows, classIdByName])

  const handleStage = () => {
    if (!targetColumn) return
    const featureColumns = headers.filter((h) => h !== targetColumn)

    onStage(
      rows.map((row, index) => {
        const featuresJson: Record<string, string | number> = {}
        for (const col of featureColumns) featuresJson[col] = coerce(row[col] ?? '')

        const targetValue = row[targetColumn]?.trim() ?? ''

        return {
          kind: 'csv' as const,
          name: `Row ${index + 1}`,
          detail: featureColumns
            .slice(0, 3)
            .map((col) => `${col}=${row[col] ?? ''}`)
            .join(', '),
          sourceName: fileName,
          split,
          classId: requiresLabelClasses ? (classIdByName.get(targetValue.toLowerCase()) ?? null) : null,
          featuresJson,
          targetValue: requiresLabelClasses ? undefined : Number(targetValue),
        }
      }),
    )

    setHeaders([])
    setRows([])
    setTargetColumn(null)
  }

  return (
    <Stack gap="md">
      {headers.length === 0 ? (
        <Dropzone onDrop={handleDrop} accept={['text/csv', '.csv']} radius="md">
          <Group justify="center" gap="md" py="xl" style={{ pointerEvents: 'none' }}>
            <CloudArrowUpIcon size={24} />
            <div>
              <Text size="sm" fw={600}>
                Drop a CSV file here or click to browse
              </Text>
              <Text size="xs" c="dimmed">
                First row must be a header row. One row per dataset item.
              </Text>
            </div>
          </Group>
        </Dropzone>
      ) : (
        <>
          <Group grow>
            <Select
              label="Target column"
              description={
                requiresLabelClasses ? 'Values must match an existing class name' : 'Numeric regression target'
              }
              placeholder="Select the label column"
              data={headers}
              value={targetColumn}
              onChange={setTargetColumn}
            />
            <Select
              label="Split"
              description="Editable per row in the preview"
              data={SPLIT_OPTIONS}
              value={split}
              onChange={(v) => setSplit((v ?? 'train') as SplitType)}
              allowDeselect={false}
            />
          </Group>

          {unmatchedValues.length > 0 && (
            <Alert icon={<WarningCircleIcon size={16} />} color="yellow" title="Unmatched class names">
              {unmatchedValues.slice(0, 10).join(', ')}
              {unmatchedValues.length > 10 ? `, +${unmatchedValues.length - 10} more` : ''} — those rows stage without a
              class. Assign one in the preview, or create the classes on the Classes page first.
            </Alert>
          )}

          <Table.ScrollContainer minWidth={400}>
            <Table striped withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  {headers.map((h) => (
                    <Table.Th key={h}>{h}</Table.Th>
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rows.slice(0, 5).map((row, i) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: static preview slice, never reordered/filtered
                  <Table.Tr key={`preview-${i}`}>
                    {headers.map((h) => (
                      <Table.Td key={h}>{row[h]}</Table.Td>
                    ))}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          <Text size="xs" c="dimmed">
            Showing {Math.min(5, rows.length)} of {rows.length} rows.
          </Text>

          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setHeaders([])}>
              Cancel
            </Button>
            <Button disabled={!targetColumn || rows.length === 0} onClick={handleStage}>
              Add {rows.length} row(s)
            </Button>
          </Group>
        </>
      )}
    </Stack>
  )
}

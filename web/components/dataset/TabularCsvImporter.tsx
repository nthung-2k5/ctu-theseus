/**
 * CSV importer for tabular tasks — the gap DataPage's old RecordEntryPanel
 * explicitly punted on: pasting raw JSON has no way to attach a target
 * label, so tabular_classification/regression snapshots built from it had a
 * null label column. This closes that loop: upload -> preview -> pick the
 * target column -> bulk-create items with both featuresJson AND the
 * matching annotation in one request.
 */

import { Alert, Button, Group, Select, Stack, Table, Text } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { CloudArrowUpIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { useEden } from '@public/lib/api'
import type { LabelClass } from '@public/store/types'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import Papa from 'papaparse'
import { useMemo, useState } from 'react'

const SPLIT_OPTIONS = [
  { value: 'train', label: 'Train' },
  { value: 'validation', label: 'Validation' },
  { value: 'test', label: 'Test' },
]

/** Best-effort scalar coercion for feature columns — numeric strings become numbers, everything else stays a string. */
function coerce(value: string): string | number {
  if (value.trim() === '') return value
  const num = Number(value)
  return Number.isFinite(num) ? num : value
}

export function TabularCsvImporter({
  projectId,
  requiresLabelClasses,
  classes,
}: {
  projectId: string
  requiresLabelClasses: boolean
  classes: LabelClass[]
}) {
  const [headers, setHeaders] = useState<string[]>([])
  const [rows, setRows] = useState<Record<string, string>[]>([])
  const [targetColumn, setTargetColumn] = useState<string | null>(null)
  const [split, setSplit] = useState<string>('train')

  const eden = useEden()
  const queryClient = useQueryClient()

  const importRows = useMutation({
    ...eden.api.projects({ projectId }).items.post.mutationOptions(),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).items.get.queryKey() })
      notifications.show({
        title: 'Import complete',
        message:
          data.failed.length > 0
            ? `${data.created.length} row(s) imported, ${data.failed.length} failed`
            : `${data.created.length} row(s) imported`,
        color: data.failed.length > 0 ? 'yellow' : 'green',
      })
      setHeaders([])
      setRows([])
      setTargetColumn(null)
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Import failed', color: 'red' })
    },
  })

  const handleDrop = (files: File[]) => {
    const file = files[0]
    if (!file) return
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

  const canImport = targetColumn && rows.length > 0 && (!requiresLabelClasses || unmatchedValues.length === 0)

  const handleImport = () => {
    if (!targetColumn) return
    const featureColumns = headers.filter((h) => h !== targetColumn)

    const items = rows.map((row) => {
      const featuresJson: Record<string, string | number> = {}
      for (const col of featureColumns) featuresJson[col] = coerce(row[col] ?? '')

      const targetValue = row[targetColumn]?.trim() ?? ''
      const annotation = requiresLabelClasses
        ? { annotationType: 'classification' as const, classId: classIdByName.get(targetValue.toLowerCase()) }
        : { annotationType: 'classification' as const, labelStructured: { value: Number(targetValue) } }

      return {
        split: split as 'train',
        tabularFeatures: { featuresJson },
        annotations: [annotation],
      }
    })

    importRows.mutate({ items })
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
              data={SPLIT_OPTIONS}
              value={split}
              onChange={(v) => setSplit(v ?? 'train')}
              allowDeselect={false}
            />
          </Group>

          {unmatchedValues.length > 0 && (
            <Alert icon={<WarningCircleIcon size={16} />} color="yellow" title="Unmatched class names">
              {unmatchedValues.slice(0, 10).join(', ')}
              {unmatchedValues.length > 10 ? `, +${unmatchedValues.length - 10} more` : ''} — create these classes on
              the Classes page first, or fix the CSV.
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
            <Button disabled={!canImport} loading={importRows.isPending} onClick={handleImport}>
              Import {rows.length} row(s)
            </Button>
          </Group>
        </>
      )}
    </Stack>
  )
}

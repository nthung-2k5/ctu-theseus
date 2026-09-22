/**
 * Export tab for a selected training run — was the standalone Models page,
 * which listed one card per succeeded run. The run is now picked in the
 * Training sidebar, so this renders the export controls for exactly one run:
 * pick an export format, dispatch the build, and download the resulting zip.
 *
 * The formats are not hardcoded here: they come from GET /api/export-formats,
 * where each one is a Python class in ai_service/theseus/export/formats/.
 */

import { Alert, Button, Card, Group, Select, Stack, Text, Title } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { DownloadSimpleIcon, PackageIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { EmptyState, StatusBadge } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  getCreateExportMutationOptions,
  getListExportFormatsQueryOptions,
  getListExportsQueryKey,
  getListExportsQueryOptions,
} from '@public/lib/api/generated/export/export'
import type { ExportFormatOut } from '@public/lib/api/generated/models'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

const EXPORT_STATUS_COLORS: Record<string, string> = {
  pending: 'gray',
  converting: 'blue',
  assembling: 'blue',
  ready: 'teal',
  failed: 'red',
}

/** Mantine grouped-select data, keeping the server's group and item order. */
function groupFormats(formats: ExportFormatOut[]) {
  const groups = new Map<string, { value: string; label: string }[]>()
  for (const f of formats) {
    const items = groups.get(f.group) ?? []
    items.push({ value: f.id, label: f.label })
    groups.set(f.group, items)
  }
  return [...groups].map(([group, items]) => ({ group, items }))
}

function ExportRow({
  modelExport,
  formatLabel,
}: {
  modelExport: {
    id: string
    status: string
    failedMessage: string | null
  }
  formatLabel: string
}) {
  return (
    <Group justify="space-between" wrap="nowrap">
      <Group gap="xs" wrap="nowrap">
        <Text size="sm">{formatLabel}</Text>
        <StatusBadge value={modelExport.status} colorMap={EXPORT_STATUS_COLORS} size="xs" />
      </Group>
      {modelExport.status === 'ready' ? (
        <Button
          size="xs"
          variant="light"
          color="teal"
          component="a"
          href={`/api/exports/${modelExport.id}/download`}
          leftSection={<DownloadSimpleIcon size={14} />}
        >
          Download
        </Button>
      ) : modelExport.status === 'failed' ? (
        <Text size="xs" c="red">
          {modelExport.failedMessage ?? 'Export failed'}
        </Text>
      ) : (
        <Text size="xs" c="dimmed">
          {modelExport.status}…
        </Text>
      )}
    </Group>
  )
}

export function RunExportPanel({ run }: { run: TrainingRunSummary }) {
  const [selectedFormat, setSelectedFormat] = useState<string | null>(null)

  const queryClient = useQueryClient()
  const exportsQueryKey = getListExportsQueryKey(run.id)

  const { data: formatsData } = useQuery({
    ...getListExportFormatsQueryOptions({ runId: run.id }),
    enabled: run.status === 'succeeded',
  })
  const formats = formatsData?.formats ?? []
  const formatData = useMemo(() => groupFormats(formats), [formats])
  // Fall back to the first format until the user picks one (or if the picked one is no longer offered).
  const format = formats.find((f) => f.id === selectedFormat) ?? formats[0]

  const { data } = useQuery({
    ...getListExportsQueryOptions(run.id),
    enabled: run.status === 'succeeded',
    refetchInterval: (query) => {
      const exports = query.state.data?.exports ?? []
      const hasPending = exports.some(
        (e) => e.status === 'pending' || e.status === 'converting' || e.status === 'assembling',
      )
      return hasPending ? 2500 : false
    },
  })
  const exports = data?.exports ?? []

  const dispatchExport = useMutation({
    ...getCreateExportMutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: exportsQueryKey })
      notifications.show({
        title: 'Export started',
        message: `Building the ${format?.label ?? 'export'} bundle…`,
        color: 'blue',
      })
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to start export'), color: 'red' })
    },
  })

  if (run.status !== 'succeeded') {
    return (
      <Card withBorder p="lg" radius="md">
        <EmptyState
          icon={PackageIcon}
          title="Nothing to export yet"
          description="This run has to finish successfully before it can be exported."
        />
      </Card>
    )
  }

  return (
    <Card withBorder p="lg" radius="md">
      <Stack gap="md">
        <div>
          <Title order={5}>Export</Title>
          <Text size="xs" c="dimmed">
            Package {run.name} as a downloadable bundle.
          </Text>
        </div>

        <Group gap="sm" align="flex-end" wrap="wrap">
          <Select
            label="Export format"
            description={format?.description}
            data={formatData}
            value={format?.id ?? null}
            onChange={(v) => setSelectedFormat(v)}
            allowDeselect={false}
            searchable
            placeholder={formats.length === 0 ? 'Loading formats…' : undefined}
            disabled={formats.length === 0}
            w={320}
          />
          <Button
            size="sm"
            loading={dispatchExport.isPending}
            disabled={!format}
            onClick={() => format && dispatchExport.mutate({ runId: run.id, data: { format: format.id } })}
          >
            Export
          </Button>
        </Group>

        {format?.notice && (
          <Alert icon={<WarningCircleIcon size={16} />} color="gray" variant="light">
            {format.notice}
          </Alert>
        )}

        {exports.length > 0 && (
          <Stack gap="xs">
            {exports.map((e) => (
              <ExportRow
                key={e.id}
                modelExport={e}
                formatLabel={formats.find((f) => f.id === e.format)?.label ?? e.format}
              />
            ))}
          </Stack>
        )}
      </Stack>
    </Card>
  )
}

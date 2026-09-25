/**
 * Export tab for a selected training run — was the standalone Models page,
 * which listed one card per succeeded run. The run is now picked in the
 * Training sidebar, so this renders the export options for exactly one run:
 * every format the run supports, each with its build status and a button to
 * build the bundle (not built / failed) or download it (ready).
 *
 * The formats are not hardcoded here: they come from GET /api/export-formats,
 * where each one is a Python class in ai_service/theseus/export/formats/.
 */

import { Badge, Button, Group, Paper, SimpleGrid, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ArrowClockwiseIcon, DownloadSimpleIcon, HammerIcon, PackageIcon } from '@phosphor-icons/react'
import { EmptyState } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  getCreateExportMutationOptions,
  getListExportFormatsQueryOptions,
  getListExportsQueryKey,
  getListExportsQueryOptions,
} from '@public/lib/api/generated/export/export'
import type { ExportFormatOut, ExportRow } from '@public/lib/api/generated/models'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

const IN_PROGRESS = ['pending', 'converting', 'assembling']

/** What a format's row shows: the server's export status, or `not_built` when no export exists yet. */
type FormatState = 'not_built' | 'building' | 'ready' | 'failed'

const STATE_BADGE: Record<FormatState, { label: string; color: string }> = {
  not_built: { label: 'Not built', color: 'gray' },
  building: { label: 'Building…', color: 'blue' },
  ready: { label: 'Ready', color: 'teal' },
  failed: { label: 'Failed', color: 'red' },
}

/**
 * The export that represents a format. Exports arrive newest-first and a format can have several
 * (a retry after a failure, or a rebuild), so prefer a ready one, then anything still running,
 * then the newest failure.
 */
function representativeExport(exports: ExportRow[]): ExportRow | undefined {
  return exports.find((e) => e.status === 'ready') ?? exports.find((e) => IN_PROGRESS.includes(e.status)) ?? exports[0]
}

function stateOf(modelExport: ExportRow | undefined): FormatState {
  if (!modelExport) return 'not_built'
  if (modelExport.status === 'ready') return 'ready'
  if (modelExport.status === 'failed') return 'failed'
  return 'building'
}

function ExportOptionRow({
  format,
  modelExport,
  building,
  onBuild,
}: {
  format: ExportFormatOut
  modelExport: ExportRow | undefined
  /** A build request for this format is in flight (the POST, before the row shows up in the list). */
  building: boolean
  onBuild: () => void
}) {
  const state = building ? 'building' : stateOf(modelExport)
  const badge = STATE_BADGE[state]
  // converting / assembling are worth naming; `pending` is just "Building…".
  const badgeLabel =
    state === 'building' && modelExport && modelExport.status !== 'pending' ? `${modelExport.status}…` : badge.label

  return (
    <Paper p="md" style={state === 'ready' ? { borderColor: 'var(--mantine-color-cyan-5)' } : undefined}>
      <Stack gap="xs" h="100%" justify="space-between">
        <div>
          <Group gap="xs" wrap="nowrap" justify="space-between">
            <Text size="sm" fw={600}>
              {format.label}
            </Text>
            <Badge color={badge.color}>{badgeLabel}</Badge>
          </Group>
          <Badge variant="outline" color="gray" mt={4}>
            {format.group}
          </Badge>
          <Text size="xs" c="dimmed" mt={6}>
            {format.description}
          </Text>
        </div>

        {state === 'ready' && modelExport ? (
          <Button
            variant="light"
            color="teal"
            component="a"
            href={`/api/exports/${modelExport.id}/download`}
            leftSection={<DownloadSimpleIcon size={14} />}
          >
            Download
          </Button>
        ) : state === 'building' ? (
          <Button variant="light" loading disabled>
            Building
          </Button>
        ) : (
          <Button
            variant={state === 'failed' ? 'light' : 'filled'}
            color={state === 'failed' ? 'red' : undefined}
            leftSection={state === 'failed' ? <ArrowClockwiseIcon size={14} /> : <HammerIcon size={14} />}
            onClick={onBuild}
          >
            {state === 'failed' ? 'Retry build' : 'Build'}
          </Button>
        )}
        {state === 'failed' && (
          <Text size="xs" c="red">
            {modelExport?.failedMessage ?? 'Export failed'}
          </Text>
        )}
      </Stack>
    </Paper>
  )
}

export function RunExportPanel({ run }: { run: TrainingRunSummary }) {
  const queryClient = useQueryClient()
  const exportsQueryKey = getListExportsQueryKey(run.id)

  const { data: formatsData } = useQuery({
    ...getListExportFormatsQueryOptions({ runId: run.id }),
    enabled: run.status === 'succeeded',
  })
  const formats = formatsData?.formats ?? []

  const { data } = useQuery({
    ...getListExportsQueryOptions(run.id),
    enabled: run.status === 'succeeded',
    refetchInterval: (query) => {
      const exports = query.state.data?.exports ?? []
      return exports.some((e) => IN_PROGRESS.includes(e.status)) ? 2500 : false
    },
  })
  const exports = data?.exports ?? []

  const dispatchExport = useMutation({
    ...getCreateExportMutationOptions(),
    // Refetch on error too: the click may have raced a build that already exists.
    onSettled: () => queryClient.invalidateQueries({ queryKey: exportsQueryKey }),
    onSuccess: (_data, { data: body }) => {
      const label = formats.find((f) => f.id === body.format)?.label ?? 'export'
      notifications.show({ title: 'Export started', message: `Building the ${label} bundle…`, color: 'blue' })
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to start export'), color: 'red' })
    },
  })
  const buildingFormat = dispatchExport.isPending ? dispatchExport.variables?.data.format : undefined

  if (run.status !== 'succeeded') {
    return (
      <EmptyState
        icon={PackageIcon}
        title="Nothing to export yet"
        description="This run has to finish successfully before it can be exported."
      />
    )
  }

  return (
    <Stack gap="sm">
      <Text size="xs" c="dimmed">
        Package {run.name} as a downloadable bundle. Build a format once, then download it any time.
      </Text>

      {formats.length === 0 && (
        <Text size="sm" c="dimmed">
          Loading export options…
        </Text>
      )}

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="sm">
        {formats.map((f) => (
          <ExportOptionRow
            key={f.id}
            format={f}
            modelExport={representativeExport(exports.filter((e) => e.format === f.id))}
            building={buildingFormat === f.id}
            onBuild={() => dispatchExport.mutate({ runId: run.id, data: { format: f.id } })}
          />
        ))}
      </SimpleGrid>
    </Stack>
  )
}

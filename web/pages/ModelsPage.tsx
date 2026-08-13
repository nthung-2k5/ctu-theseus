/**
 * Models page – export succeeded training runs to a deployable format and
 * download the artifact once the worker finishes writing it.
 */

import { Badge, Button, Card, Group, Loader, Select, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { DownloadSimpleIcon, PackageIcon } from '@phosphor-icons/react'
import { api } from '@public/lib/api'
import { useEdenMutation } from '@public/lib/eden-query'
import { useTrainingRuns } from '@public/queries/training'
import type { TrainingRunSummary } from '@public/store/types'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useParams } from 'wouter'

type ExportFormat = 'onnx' | 'torchscript'
const FORMAT_OPTIONS = [
  { value: 'onnx', label: 'ONNX' },
  { value: 'torchscript', label: 'TorchScript' },
]

function RunExportCard({ run }: { run: TrainingRunSummary }) {
  const [format, setFormat] = useState<ExportFormat>('onnx')
  const [dispatched, setDispatched] = useState(false)

  const exportRun = useEdenMutation((f: ExportFormat) => api.runs({ runId: run.id }).export.post({ format: f }), [], {
    onSuccess: () => {
      setDispatched(true)
      notifications.show({ title: 'Export started', message: `Exporting to ${format.toUpperCase()}…`, color: 'blue' })
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Failed to start export', color: 'red' })
    },
  })

  const {
    data: readyData,
    refetch: checkReady,
    isFetching: checking,
  } = useQuery({
    queryKey: ['export-ready', run.id, format],
    queryFn: async () => {
      const { data, error } = await api.runs({ runId: run.id }).export({ format }).get()
      if (error) throw error
      return data
    },
    enabled: false,
  })

  return (
    <Card withBorder p="lg" radius="md">
      <Group justify="space-between" mb="sm">
        <div>
          <Text fw={600}>{run.name}</Text>
          <Text size="xs" c="dimmed">
            Trained {new Date(run.createdAt).toLocaleDateString()}
          </Text>
        </div>
        <Badge variant="light" color="green" size="sm">
          Succeeded
        </Badge>
      </Group>

      <Group gap="sm" align="flex-end">
        <Select
          label="Format"
          data={FORMAT_OPTIONS}
          value={format}
          onChange={(v) => {
            setFormat((v as ExportFormat) ?? 'onnx')
            setDispatched(false)
          }}
          allowDeselect={false}
          w={160}
        />
        <Button size="sm" loading={exportRun.isPending} onClick={() => exportRun.mutate(format)}>
          Export
        </Button>
        {dispatched && (
          <Button size="sm" variant="light" loading={checking} onClick={() => checkReady()}>
            Check status
          </Button>
        )}
        {readyData?.ready && (
          <Button
            size="sm"
            variant="light"
            color="teal"
            component="a"
            href={`/api/runs/${run.id}/download/${format}`}
            leftSection={<DownloadSimpleIcon size={14} />}
          >
            Download
          </Button>
        )}
      </Group>
      {dispatched && readyData && !readyData.ready && (
        <Text size="xs" c="dimmed" mt="xs">
          Still exporting — check again in a moment.
        </Text>
      )}
    </Card>
  )
}

export function ModelsPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id

  const { data, isLoading } = useTrainingRuns(projectId)
  const succeededRuns = (data?.runs ?? []).filter((r: TrainingRunSummary) => r.status === 'succeeded')

  return (
    <Stack gap="xl">
      <div>
        <Title order={2}>Models</Title>
        <Text size="sm" c="dimmed" mt={4}>
          Export successfully trained runs and download the resulting artifact.
        </Text>
      </div>

      {isLoading ? (
        <Card withBorder p="xl" radius="md" ta="center">
          <Loader size="sm" />
        </Card>
      ) : succeededRuns.length === 0 ? (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="sm">
            <ThemeIcon size={48} variant="light" color="gray" radius="xl">
              <PackageIcon size={26} weight="thin" />
            </ThemeIcon>
            <Text size="sm" c="dimmed">
              No successfully trained runs yet.
            </Text>
          </Stack>
        </Card>
      ) : (
        <Stack gap="md">
          {succeededRuns.map((run: TrainingRunSummary) => (
            <RunExportCard key={run.id} run={run} />
          ))}
        </Stack>
      )}
    </Stack>
  )
}

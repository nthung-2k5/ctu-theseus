/**
 * Export tab for a selected training run — was the standalone Models page,
 * which listed one card per succeeded run. The run is now picked in the
 * Training sidebar, so this renders the export controls for exactly one run:
 * pick a tier (model / devkit / app — see server/lib/export/bundle.ts),
 * dispatch the build, and download the resulting zip.
 */

import { Alert, Button, Card, Group, Select, Stack, Text, Title } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { DownloadSimpleIcon, PackageIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { EmptyState, StatusBadge } from '@public/components/ui'
import { useEden } from '@public/lib/api'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

type ExportTier = 'model' | 'devkit' | 'app'
type ExportFormat = 'onnx' | 'torchscript'
type ExportLang = 'python' | 'typescript' | 'csharp' | 'java' | 'pwa' | 'flutter'

const TIER_OPTIONS = [
  { value: 'model', label: 'Model only' },
  { value: 'devkit', label: 'Devkit (client source)' },
  { value: 'app', label: 'App (PWA or Flutter — no server)' },
]
const FORMAT_OPTIONS = [
  { value: 'onnx', label: 'ONNX' },
  { value: 'torchscript', label: 'TorchScript' },
]
const DEVKIT_LANG_OPTIONS = [
  { value: 'python', label: 'Python' },
  { value: 'typescript', label: 'TypeScript' },
  { value: 'csharp', label: 'C#' },
  { value: 'java', label: 'Java / Kotlin' },
]
const APP_TARGET_OPTIONS = [
  { value: 'pwa', label: 'Progressive Web App' },
  { value: 'flutter', label: 'Flutter' },
]

const EXPORT_STATUS_COLORS: Record<string, string> = {
  pending: 'gray',
  converting: 'blue',
  assembling: 'blue',
  ready: 'teal',
  failed: 'red',
}

function ExportRow({
  modelExport,
}: {
  modelExport: {
    id: string
    tier: string
    format: string
    lang: string | null
    status: string
    failedMessage: string | null
  }
}) {
  const label = [modelExport.tier, modelExport.format, modelExport.lang].filter(Boolean).join(' · ')
  return (
    <Group justify="space-between" wrap="nowrap">
      <Group gap="xs" wrap="nowrap">
        <Text size="sm">{label}</Text>
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
  const [tier, setTier] = useState<ExportTier>('model')
  const [format, setFormat] = useState<ExportFormat>('onnx')
  const [lang, setLang] = useState<ExportLang>('python')

  const eden = useEden()
  const queryClient = useQueryClient()
  const exportsQueryKey = eden.api.runs({ runId: run.id }).exports.get.queryKey()

  const { data } = useQuery({
    ...eden.api.runs({ runId: run.id }).exports.get.queryOptions(),
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
    ...eden.api.runs({ runId: run.id }).exports.post.mutationOptions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: exportsQueryKey })
      notifications.show({ title: 'Export started', message: `Building the ${tier} bundle…`, color: 'blue' })
    },
    onError: (error) => {
      const value: unknown = error.value
      const message = typeof value === 'string' ? value : (value as { message?: string } | undefined)?.message
      notifications.show({ title: 'Error', message: message ?? 'Failed to start export', color: 'red' })
    },
  })

  const isTiered = tier !== 'model'

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
            label="Tier"
            data={TIER_OPTIONS}
            value={tier}
            onChange={(v) => {
              const nextTier = (v as ExportTier) ?? 'model'
              setTier(nextTier)
              if (nextTier === 'devkit') setLang('python')
              else if (nextTier === 'app') setLang('pwa')
            }}
            allowDeselect={false}
            w={200}
          />
          <Select
            label="Format"
            data={isTiered ? FORMAT_OPTIONS.filter((f) => f.value === 'onnx') : FORMAT_OPTIONS}
            value={isTiered ? 'onnx' : format}
            onChange={(v) => setFormat((v as ExportFormat) ?? 'onnx')}
            disabled={isTiered}
            allowDeselect={false}
            w={140}
          />
          {tier === 'devkit' && (
            <Select
              label="Language"
              data={DEVKIT_LANG_OPTIONS}
              value={lang}
              onChange={(v) => setLang((v as ExportLang) ?? 'python')}
              allowDeselect={false}
              w={160}
            />
          )}
          {tier === 'app' && (
            <Select
              label="Target"
              data={APP_TARGET_OPTIONS}
              value={lang}
              onChange={(v) => setLang((v as ExportLang) ?? 'pwa')}
              allowDeselect={false}
              w={200}
            />
          )}
          <Button
            size="sm"
            loading={dispatchExport.isPending}
            onClick={() =>
              dispatchExport.mutate({ tier, format: isTiered ? 'onnx' : format, lang: isTiered ? lang : undefined })
            }
          >
            Export
          </Button>
        </Group>

        {isTiered && (
          <Alert icon={<WarningCircleIcon size={16} />} color="gray" variant="light">
            {tier === 'devkit'
              ? 'Devkit bundles ship source only (no project/build files — dependencies are documented in the README) and only support ONNX.'
              : 'App bundles are a Progressive Web App or Flutter app that runs inference on-device — never a server — and only support ONNX.'}{' '}
            Preprocessing is currently implemented for image classification only — see the bundle's README for other
            modalities.
          </Alert>
        )}

        {exports.length > 0 && (
          <Stack gap="xs">
            {exports.map((e) => (
              <ExportRow key={e.id} modelExport={e} />
            ))}
          </Stack>
        )}
      </Stack>
    </Card>
  )
}

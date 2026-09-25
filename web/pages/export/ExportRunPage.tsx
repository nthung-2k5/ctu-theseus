import { ActionIcon, Code, CopyButton, Paper, SimpleGrid, Stack, Tabs, Text, Tooltip } from '@mantine/core'
import { CheckIcon, CopyIcon } from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { RunExportPanel } from '@public/components/training/RunExportPanel'
import {
  CopyField,
  EmptyState,
  LinkButton,
  PageHeader,
  QueryBoundary,
  SectionLabel,
  StatusBadge,
} from '@public/components/ui'
import { formatDateTime } from '@public/lib/format'
import { projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import { getInferenceInputSpec, getTaskDescriptor } from '@public/lib/tasks'
import { MONO_STACK } from '@public/theme'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { predictSnippets } from './snippets'

const routeApi = getRouteApi('/_app/project/$projectId/export/$runId')

function Snippet({ code }: { code: string }) {
  return (
    <div style={{ position: 'relative' }}>
      <Code block style={{ fontFamily: MONO_STACK, fontSize: 12 }}>
        {code}
      </Code>
      <CopyButton value={code} timeout={1500}>
        {({ copied, copy }) => (
          <Tooltip label={copied ? 'Copied' : 'Copy'}>
            <ActionIcon
              variant="subtle"
              color={copied ? 'teal' : 'gray'}
              onClick={copy}
              aria-label="Copy snippet"
              style={{ position: 'absolute', top: 6, right: 6 }}
            >
              {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
    </div>
  )
}

export function ExportRunPage() {
  const { projectId, runId } = routeApi.useParams()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const { data, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const run = data?.runs.find((r) => r.id === runId)

  if (!run) {
    return (
      <div className="flex flex-col gap-3 p-3">
        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          <EmptyState title="Run not found" description="It may have been deleted." />
        </QueryBoundary>
      </div>
    )
  }

  const descriptor = getTaskDescriptor(project.task)
  const snapshot = project.dataset?.versions?.find((v) => v.id === run.datasetVersionId)
  const evaluation = run.evaluation?.status === 'success' ? run.evaluation : null
  const snippets = predictSnippets(runId, getInferenceInputSpec(project.task), window.location.origin)

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title={`Export · ${run.name}`}
        badges={<StatusBadge value={run.status} colorMap={STATUS_COLORS} />}
        description="Build a bundle to download, or call the model over the prediction API."
        actions={
          <LinkButton to="/project/$projectId/export" params={{ projectId }} variant="default">
            All models
          </LinkButton>
        }
      />

      <Paper p="md">
        <Stack gap="xs">
          <SectionLabel>Model card</SectionLabel>
          <SimpleGrid cols={{ base: 2, md: 5 }} spacing="sm">
            <Fact label="Task" value={descriptor.label} />
            <Fact label="Snapshot" value={snapshot?.versionTag ?? run.datasetVersionId.slice(0, 8)} />
            <Fact label="Accuracy" value={evaluation?.accuracy != null ? evaluation.accuracy.toFixed(3) : '—'} />
            <Fact label="Macro F1" value={evaluation?.macroF1 != null ? evaluation.macroF1.toFixed(3) : '—'} />
            <Fact label="Trained" value={run.completedAt ? formatDateTime(run.completedAt) : '—'} />
          </SimpleGrid>
          <div>
            <Text size="xs" c="dimmed">
              Run id
            </Text>
            <CopyField value={run.id} truncate={false} />
          </div>
        </Stack>
      </Paper>

      <div>
        <SectionLabel mb={6}>Formats</SectionLabel>
        <RunExportPanel run={run} />
      </div>

      <Paper p="md">
        <Stack gap="xs">
          <SectionLabel>Use it: prediction API</SectionLabel>
          <Text size="xs" c="dimmed">
            Authenticate with an API key (create one under Settings) sent as a bearer token. The response is the same
            prediction the playground shows.
          </Text>
          <Tabs defaultValue="curl">
            <Tabs.List>
              <Tabs.Tab value="curl">curl</Tabs.Tab>
              <Tabs.Tab value="python">Python</Tabs.Tab>
            </Tabs.List>
            <Tabs.Panel value="curl" pt="xs">
              <Snippet code={snippets.curl} />
            </Tabs.Panel>
            <Tabs.Panel value="python" pt="xs">
              <Snippet code={snippets.python} />
            </Tabs.Panel>
          </Tabs>
        </Stack>
      </Paper>
    </div>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text size="sm" fw={500} className="tnum">
        {value}
      </Text>
    </div>
  )
}

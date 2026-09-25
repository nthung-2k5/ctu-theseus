import { Text } from '@mantine/core'
import { ExportIcon } from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import {
  DataTable,
  type DataTableColumn,
  EmptyState,
  LinkButton,
  PageHeader,
  QueryBoundary,
  StatusBadge,
} from '@public/components/ui'
import { formatDate } from '@public/lib/format'
import { useTrainingRuns } from '@public/lib/queries'
import type { TrainingRunSummary } from '@public/store/types'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/export/')

/** Every finished run is a candidate for packaging. */
export function ExportPage() {
  const { projectId } = routeApi.useParams()
  const navigate = routeApi.useNavigate()
  const { data, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const runs = (data?.runs ?? []).filter((r) => r.status === 'succeeded')

  const columns: DataTableColumn<TrainingRunSummary>[] = [
    {
      key: 'name',
      header: 'Run',
      render: (run) => (
        <div>
          <Text size="sm" fw={500}>
            {run.name}
          </Text>
          <Text size="xs" c="dimmed" className="tnum">
            {run.id.slice(0, 8)}
          </Text>
        </div>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (run) => <StatusBadge value={run.status} colorMap={STATUS_COLORS} />,
    },
    {
      key: 'accuracy',
      header: 'Accuracy',
      fit: true,
      render: (run) => (
        <Text size="sm" className="tnum">
          {run.evaluation?.status === 'success' && run.evaluation.accuracy != null
            ? run.evaluation.accuracy.toFixed(3)
            : '—'}
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (run) => (
        <Text size="xs" c="dimmed" className="tnum">
          {formatDate(run.createdAt)}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (run) => (
        <LinkButton
          to="/project/$projectId/export/$runId"
          params={{ projectId, runId: run.id }}
          size="compact-sm"
          variant="light"
          leftSection={<ExportIcon size={14} />}
        >
          Package
        </LinkButton>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader title="Export" description="Package a trained model and call it from your own code." />
      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {runs.length === 0 ? (
          <EmptyState
            icon={ExportIcon}
            title="Nothing to export yet"
            description="A run has to finish successfully before it can be exported."
          />
        ) : (
          <DataTable
            columns={columns}
            data={runs}
            getRowKey={(run) => run.id}
            onRowClick={(run) =>
              navigate({ to: '/project/$projectId/export/$runId', params: { projectId, runId: run.id } })
            }
          />
        )}
      </QueryBoundary>
    </div>
  )
}

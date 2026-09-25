import { Badge, Group, SimpleGrid, Text } from '@mantine/core'
import { PlayIcon } from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { EmptyState, LinkButton, LinkCard, PageHeader, QueryBoundary, StatusBadge } from '@public/components/ui'
import { formatDate } from '@public/lib/format'
import { useTrainingRuns } from '@public/lib/queries'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/playground/')

/** Pick a finished run to try: one card per succeeded run. */
export function PlaygroundPage() {
  const { projectId } = routeApi.useParams()
  const { data, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const runs = (data?.runs ?? []).filter((r) => r.status === 'succeeded')

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader title="Playground" description="Run predictions with a trained model on real inputs." />

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {runs.length === 0 ? (
          <EmptyState
            icon={PlayIcon}
            title="No trained models yet"
            description="A run has to finish successfully before you can test it."
            action={
              <LinkButton to="/project/$projectId/experiments" params={{ projectId }} variant="default">
                Go to experiments
              </LinkButton>
            }
          />
        ) : (
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="sm">
            {runs.map((run) => (
              <LinkCard
                key={run.id}
                to="/project/$projectId/playground/$runId"
                params={{ projectId, runId: run.id }}
                withBorder
                padding="md"
                className="card-elevated"
                style={{ textDecoration: 'none', color: 'inherit', display: 'flex', flexDirection: 'column', gap: 6 }}
              >
                <Group justify="space-between" wrap="nowrap">
                  <Text fw={600} truncate>
                    {run.name}
                  </Text>
                  <StatusBadge value={run.status} colorMap={STATUS_COLORS} />
                </Group>
                <Group gap={6}>
                  {run.evaluation?.status === 'success' && run.evaluation.accuracy != null && (
                    <Badge color="teal">acc {run.evaluation.accuracy.toFixed(3)}</Badge>
                  )}
                </Group>
                <Text size="xs" c="dimmed" className="tnum">
                  {run.id.slice(0, 8)} · {formatDate(run.createdAt)}
                </Text>
              </LinkCard>
            ))}
          </SimpleGrid>
        )}
      </QueryBoundary>
    </div>
  )
}

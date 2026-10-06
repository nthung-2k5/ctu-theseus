import { Button, SimpleGrid } from '@mantine/core'
import { StatCard } from '@public/components/ui'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/classes')

/**
 * The dataset's label coverage at a glance: how many classes, how much of the draft is labelled, what is not,
 * and how many classes need a look. Both of the last two are actions: one opens the unlabelled items on the
 * Dataset page, the other narrows this table to the classes with a problem.
 */
export function ClassStatsStrip({
  classCount,
  total,
  unassigned,
  retired,
  attentionCount,
  attentionOnly,
  onToggleAttention,
}: {
  classCount: number
  total: number
  unassigned: number
  retired: number
  attentionCount: number
  attentionOnly: boolean
  onToggleAttention: () => void
}) {
  const navigate = routeApi.useNavigate()
  const { projectId } = routeApi.useParams()
  const unlabeled = unassigned + retired
  const labeledPct = total > 0 ? Math.round(((total - unlabeled) / total) * 100) : null

  return (
    <SimpleGrid cols={{ base: 2, md: 4 }} spacing="sm">
      <StatCard label="Classes" value={classCount} />
      <StatCard
        label="Labelled"
        value={labeledPct === null ? '—' : `${labeledPct}%`}
        hint={`${(total - unlabeled).toLocaleString()} of ${total.toLocaleString()} draft items`}
      />
      <StatCard
        label="Unlabelled"
        value={unlabeled.toLocaleString()}
        hint={
          unlabeled > 0 ? (
            <Button
              variant="subtle"
              size="compact-xs"
              px={0}
              onClick={() =>
                void navigate({
                  to: '/project/$projectId/dataset',
                  params: { projectId },
                  search: { page: 1, perPage: 20, classId: 'unassigned' },
                })
              }
            >
              {retired > 0 ? `Show the ${unassigned.toLocaleString()} with no class` : 'Show these items'}
            </Button>
          ) : (
            'Every item has a class'
          )
        }
      />
      <StatCard
        label="Need attention"
        value={attentionCount}
        hint={
          attentionCount > 0 ? (
            <Button variant="subtle" size="compact-xs" px={0} onClick={onToggleAttention}>
              {attentionOnly ? 'Show all classes' : 'Show only these'}
            </Button>
          ) : (
            'No empty or tiny classes'
          )
        }
      />
    </SimpleGrid>
  )
}

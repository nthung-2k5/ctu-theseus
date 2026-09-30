/**
 * Admin > Overview: how much there is, how busy each job lane is, and whether each trainer backend can run.
 * Read from Postgres and the in-process dispatcher only (never from object storage), so it stays cheap
 * however much data there is. It refreshes on its own so a busy lane can be watched.
 */

import { Badge, Group, Paper, SimpleGrid, Stack, Text } from '@mantine/core'
import { BrainIcon, CubeIcon, ExportIcon, FoldersIcon, ShieldCheckIcon, UsersIcon } from '@phosphor-icons/react'
import { MeterBar, PageHeader, QueryBoundary, SectionLabel, StatCard } from '@public/components/ui'
import { getAdminSystemQueryOptions } from '@public/lib/api/generated/admin/admin'
import { formatBytes } from '@public/lib/customModels'
import { useQuery } from '@tanstack/react-query'

export function OverviewPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    ...getAdminSystemQueryOptions(),
    refetchInterval: 5000,
  })
  const counts = data?.counts

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Overview" description="The platform at a glance." />
      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {data && counts && (
          <>
            <SimpleGrid cols={{ base: 2, md: 3, lg: 6 }} spacing="sm">
              <StatCard
                icon={UsersIcon}
                label="Users"
                value={counts.users}
                hint={`${counts.admins} admin${counts.admins === 1 ? '' : 's'}${counts.disabledUsers ? `, ${counts.disabledUsers} disabled` : ''}`}
              />
              <StatCard icon={FoldersIcon} label="Projects" value={counts.projects} />
              <StatCard
                icon={BrainIcon}
                label="Training runs"
                value={counts.runs}
                hint={`${counts.activeRuns} active`}
              />
              <StatCard icon={ExportIcon} label="Exports" value={counts.exports} />
              <StatCard
                icon={CubeIcon}
                label="Custom models"
                value={counts.customModels}
                hint={formatBytes(counts.customModelBytes)}
              />
              <StatCard
                icon={ShieldCheckIcon}
                label="Backends"
                value={data.backends.filter((b) => b.available).length}
                hint={`of ${data.backends.length} usable`}
              />
            </SimpleGrid>

            <Paper p="md">
              <SectionLabel mb="xs">Job lanes</SectionLabel>
              <Stack gap="sm">
                {data.lanes.map((lane) => (
                  <div key={lane.name}>
                    <Group justify="space-between">
                      <Text size="sm" fw={600}>
                        {lane.label}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {lane.running != null && lane.capacity != null
                          ? `${lane.running} of ${lane.capacity} running`
                          : 'not running'}
                        {' · '}
                        {lane.queued} waiting
                      </Text>
                    </Group>
                    <MeterBar value={lane.running ?? 0} max={lane.capacity ?? 1} />
                  </div>
                ))}
              </Stack>
            </Paper>

            <Paper p="md">
              <SectionLabel mb="xs">Trainer backends</SectionLabel>
              <Stack gap={6}>
                {data.backends.map((b) => (
                  <Group key={b.id} justify="space-between" wrap="nowrap">
                    <div>
                      <Text size="sm" fw={600}>
                        {b.label}
                      </Text>
                      {b.unavailableReason && (
                        <Text size="xs" c="dimmed">
                          {b.unavailableReason}
                        </Text>
                      )}
                    </div>
                    <Badge variant="light" color={b.available ? 'teal' : 'red'}>
                      {b.available ? 'available' : 'unavailable'}
                    </Badge>
                  </Group>
                ))}
              </Stack>
            </Paper>
          </>
        )}
      </QueryBoundary>
    </div>
  )
}

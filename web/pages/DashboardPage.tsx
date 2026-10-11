import { Badge, Button, Group, Paper, SimpleGrid, Skeleton, Stack, Text } from '@mantine/core'
import { PlusIcon } from '@phosphor-icons/react'
import { openCreateProjectModal } from '@public/components/CreateProjectModal'
import { STATUS_COLORS } from '@public/components/training/constants'
import { LinkCard, PageHeader, QueryBoundary, StatusBadge } from '@public/components/ui'
import type { ProjectTask } from '@public/lib/api/enums'
import { formatDate } from '@public/lib/format'
import { MODALITY_META } from '@public/lib/modality'
import { useProjects } from '@public/lib/queries'
import { taskRegistry } from '@public/lib/tasks'

function taskLabel(task: string): string {
  return taskRegistry[task as ProjectTask]?.label ?? task
}

export function DashboardPage() {
  const { data, isLoading, isError, refetch } = useProjects()
  const projects = [...(data?.projects ?? [])].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  )

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Projects"
        actions={
          <Button leftSection={<PlusIcon size={15} />} onClick={openCreateProjectModal}>
            New project
          </Button>
        }
      />

      <QueryBoundary
        isLoading={isLoading}
        isError={isError}
        onRetry={() => refetch()}
        loadingFallback={
          <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }} spacing="sm">
            {Array.from({ length: 6 }, (_, i) => `skeleton-${i}`).map((key) => (
              <Paper key={key} p="md">
                <Stack gap="xs">
                  <Skeleton height={16} width="50%" radius="sm" />
                  <Skeleton height={12} width="90%" radius="sm" />
                  <Skeleton height={12} width="60%" radius="sm" />
                </Stack>
              </Paper>
            ))}
          </SimpleGrid>
        }
      >
        {projects.length === 0 && (
          <Text c="dimmed" ta="center" py="xl">
            No projects yet. Create one to get started.
          </Text>
        )}
        <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }} spacing="sm">
          {projects.map((project) => {
            const modality = taskRegistry[project.task]?.modality ?? 'vision'
            const meta = MODALITY_META[modality] ?? MODALITY_META.vision
            return (
              <LinkCard
                key={project.id}
                to="/project/$projectId"
                params={{ projectId: project.id }}
                withBorder
                padding="md"
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <Group justify="space-between" mb={4} wrap="nowrap">
                  <Text fw={600} truncate>
                    {project.name}
                  </Text>
                  <Badge color={meta.color}>{meta.label}</Badge>
                </Group>
                <Badge variant="outline" color="gray" mb="xs">
                  {taskLabel(project.task)}
                </Badge>
                <Text size="xs" c="dimmed" lineClamp={2} mih={32}>
                  {project.description || 'No description'}
                </Text>
                <Group justify="space-between" mt="sm" wrap="nowrap" gap="xs">
                  <Text size="xs" c="dimmed" className="tnum">
                    {(project.itemCount ?? 0).toLocaleString()} item{project.itemCount === 1 ? '' : 's'} ·{' '}
                    {project.runCount ?? 0} run{project.runCount === 1 ? '' : 's'}
                  </Text>
                  {project.lastRunStatus && <StatusBadge value={project.lastRunStatus} colorMap={STATUS_COLORS} />}
                </Group>
                <Text size="xs" c="dimmed" className="tnum" mt={2}>
                  Active {formatDate(project.lastActivityAt ?? project.createdAt)}
                </Text>
              </LinkCard>
            )
          })}
        </SimpleGrid>
      </QueryBoundary>
    </div>
  )
}

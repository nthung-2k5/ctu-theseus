import { ActionIcon, Badge, Box, Card, Group, SimpleGrid, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import { modals } from '@mantine/modals'
import {
  ArrowRightIcon,
  BrainIcon,
  CrosshairIcon,
  DatabaseIcon,
  PencilSimpleIcon,
  StackIcon,
  TagIcon,
} from '@phosphor-icons/react'
import { UpdateProjectModal } from '@public/components/UpdateProjectModal'
import { StatCard } from '@public/components/ui'
import { MODALITY_COLORS } from '@public/lib/constants'
import { projectDetailQueryOptions } from '@public/lib/queries'
import { isClassificationTask } from '@server/lib/tasks'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, useNavigate } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/')

const BASE_WORKFLOW_STEPS = [
  {
    label: 'Data',
    desc: 'Upload items and build a labeled pool for training',
    icon: DatabaseIcon,
    to: '/project/$projectId/data' as const,
    color: 'primary',
  },
  {
    label: 'Dataset',
    desc: 'Assign splits and create version snapshots for training',
    icon: StackIcon,
    to: '/project/$projectId/dataset' as const,
    color: 'blue',
  },
  {
    label: 'Training',
    desc: 'Train models on dataset versions and monitor progress',
    icon: BrainIcon,
    to: '/project/$projectId/training' as const,
    color: 'teal',
  },
  {
    label: 'Inference',
    desc: 'Test trained models and export weights for deployment',
    icon: CrosshairIcon,
    to: '/project/$projectId/inference' as const,
    color: 'orange',
  },
]

const CLASSES_STEP = {
  label: 'Classes',
  desc: 'Define label classes for classification annotations',
  icon: TagIcon,
  to: '/project/$projectId/classes' as const,
  color: 'violet',
}

export function ProjectPage() {
  const { projectId } = routeApi.useParams()
  const navigate = useNavigate()
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const dataset = activeProject.dataset
  const versionCount = (dataset?.versions?.length ?? 0) + (dataset?.draft ? 1 : 0)

  const WORKFLOW_STEPS = isClassificationTask(activeProject.task)
    ? [BASE_WORKFLOW_STEPS[0], CLASSES_STEP, ...BASE_WORKFLOW_STEPS.slice(1)]
    : BASE_WORKFLOW_STEPS

  const handleEditClick = () => {
    modals.open({
      title: 'Edit project',
      children: (
        <UpdateProjectModal
          projectId={activeProject.id}
          projectName={activeProject.name}
          projectDescription={activeProject.description}
        />
      ),
    })
  }

  return (
    <Box>
      <Stack gap="xl">
        {/* Header */}
        <div>
          <Group gap="xs">
            <Title order={2}>{activeProject.name}</Title>
            <ActionIcon variant="subtle" color="gray" onClick={handleEditClick} size="lg">
              <PencilSimpleIcon size={20} />
            </ActionIcon>
          </Group>
          <Text size="sm" c="dimmed" mt={4}>
            {activeProject.description || 'Project overview and workflow'}
          </Text>
          <Group gap="xs" mt="xs">
            <Badge variant="light" size="sm">
              {activeProject.task.replace(/_/g, ' ')}
            </Badge>
            {dataset && (
              <Badge variant="light" color={MODALITY_COLORS[dataset.modality]} size="sm">
                {dataset.modality}
              </Badge>
            )}
          </Group>
        </div>

        {/* Stats */}
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="lg">
          <StatCard icon={DatabaseIcon} color="primary" label="Dataset Versions" value={versionCount} />
          <StatCard icon={BrainIcon} color="teal" label="Training Runs" value={activeProject.runCount ?? 0} />
          <StatCard
            icon={StackIcon}
            color="violet"
            label="Modality"
            value={<Text tt="capitalize">{dataset?.modality ?? '—'}</Text>}
          />
        </SimpleGrid>

        {/* Workflow */}
        <div>
          <Title order={4} mb="md">
            Workflow
          </Title>
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
            {WORKFLOW_STEPS.map((step, i) => (
              <Card
                key={step.label}
                withBorder
                padding="lg"
                radius="md"
                className="card-elevated"
                style={{ cursor: 'pointer' }}
                onClick={() => navigate({ to: step.to, params: { projectId } })}
              >
                <Group justify="space-between" mb="sm">
                  <Group gap="sm">
                    <ThemeIcon size="md" variant="light" color={step.color}>
                      <step.icon size={18} />
                    </ThemeIcon>
                    <Text fw={600}>{step.label}</Text>
                  </Group>
                  <Badge variant="dot" color={step.color} size="sm">
                    Step {i + 1}
                  </Badge>
                </Group>
                <Text size="sm" c="dimmed">
                  {step.desc}
                </Text>
                <Group justify="flex-end" mt="sm">
                  <ArrowRightIcon size={16} />
                </Group>
              </Card>
            ))}
          </SimpleGrid>
        </div>
      </Stack>
    </Box>
  )
}

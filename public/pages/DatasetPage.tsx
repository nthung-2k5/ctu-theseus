import {
  Badge,
  Box,
  Button,
  Card,
  Group,
  Modal,
  Pagination,
  SimpleGrid,
  Skeleton,
  Stack,
  Table,
  Text,
  TextInput,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { DatabaseIcon, PlusIcon, StackIcon } from '@phosphor-icons/react'
import { api } from '@public/lib/api'
import { MODALITY_COLORS, SPLIT_COLORS } from '@public/lib/constants'
import { useEdenMutation } from '@public/lib/eden-query'
import { queries } from '@public/queries'
import { useProjectItems } from '@public/queries/dataset'
import type { DatasetVersion } from '@public/store/types'
import { useProjectStore } from '@public/store/useProjectStore'
import { useState } from 'react'
import { useParams } from 'wouter'

/* ── Create Version Modal ── */
const CreateVersionModal = ({ projectId, onClose }: { projectId: string; onClose: () => void }) => {
  const form = useForm({
    initialValues: { versionTag: '' },
    validate: {
      versionTag: (v) => (v.trim().length > 0 ? null : 'Version tag is required'),
    },
  })

  const createVersion = useEdenMutation(
    (body: any) => api.projects({ projectId }).versions.post(body),
    [queries.projects.detail(projectId)._ctx.summary.queryKey],
    {
      onSuccess: () => {
        notifications.show({ title: 'Version created', message: 'New snapshot version created', color: 'green' })
        form.reset()
        onClose()
      },
      onError: () => {
        notifications.show({ title: 'Error', message: 'Failed to create version', color: 'red' })
      },
    },
  )

  return (
    <form onSubmit={form.onSubmit((values) => createVersion.mutate(values as any))}>
      <Stack gap="md">
        <TextInput
          label="Version tag"
          placeholder="e.g. v1.0, snapshot-2024-01"
          {...form.getInputProps('versionTag')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" loading={createVersion.isPending}>
            Create Snapshot
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

/* ── Split Items Viewer ── */
const SplitItemsPanel = ({
  projectId,
  versionId,
  splitType,
}: {
  projectId: string
  versionId: string
  splitType: 'train' | 'validation' | 'test'
}) => {
  const [page, setPage] = useState(1)
  const perPage = 20

  const { data, isLoading } = useProjectItems(projectId, { versionId, split: splitType, page, perPage })
  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text size="sm" fw={600} tt="capitalize">
          {splitType} Split
        </Text>
        <Badge variant="light" size="sm">
          {total} items
        </Badge>
      </Group>

      {isLoading ? (
        <Stack gap="xs">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} height={40} radius="sm" />
          ))}
        </Stack>
      ) : items.length === 0 ? (
        <Card withBorder p="lg" radius="md" ta="center">
          <Text size="sm" c="dimmed">
            No items in this split yet
          </Text>
        </Card>
      ) : (
        <>
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>ID</Table.Th>
                <Table.Th>External ID</Table.Th>
                <Table.Th>Features</Table.Th>
                <Table.Th>Annotations</Table.Th>
                <Table.Th>Created</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {items.map((item) => (
                <Table.Tr key={item.id}>
                  <Table.Td>
                    <Text size="xs" ff="monospace">
                      {item.id.slice(0, 8)}…
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs">{item.externalId ?? '—'}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Group gap={4}>
                      {item.textFeatures && (
                        <Badge size="xs" color="blue">
                          text
                        </Badge>
                      )}
                      {item.visionFeatures && (
                        <Badge size="xs" color="green">
                          vision
                        </Badge>
                      )}
                      {item.audioFeatures && (
                        <Badge size="xs" color="orange">
                          audio
                        </Badge>
                      )}
                      {item.tabularFeatures && (
                        <Badge size="xs" color="grape">
                          tabular
                        </Badge>
                      )}
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Badge size="xs" variant="light">
                      {item.annotations?.length ?? 0}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {new Date(item.createdAt).toLocaleDateString()}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>

          {totalPages > 1 && (
            <Group justify="center">
              <Pagination total={totalPages} value={page} onChange={setPage} size="sm" />
            </Group>
          )}
        </>
      )}
    </Stack>
  )
}

const SPLIT_TYPES = ['train', 'validation', 'test'] as const

/* ── Version Card ── */
const VersionCard = ({
  version,
  isDraft,
  onSelectSplit,
}: {
  version: DatasetVersion
  isDraft: boolean
  onSelectSplit: (versionId: string, splitType: (typeof SPLIT_TYPES)[number]) => void
}) => {
  const counts = Object.fromEntries(SPLIT_TYPES.map((s) => [s, 0])) as Record<(typeof SPLIT_TYPES)[number], number>
  for (const item of version.items ?? []) counts[item.splitType]++

  return (
    <Card withBorder p="lg" radius="md">
      <Group justify="space-between" mb="md">
        <Group gap="sm">
          <ThemeIcon size="md" variant="light" color={isDraft ? 'blue' : 'teal'}>
            <StackIcon size={16} />
          </ThemeIcon>
          <Title order={5}>{isDraft ? 'Draft' : version.versionTag}</Title>
          <Badge variant="light" color={isDraft ? 'blue' : 'teal'} size="sm">
            {isDraft ? 'Working Copy' : 'Snapshot'}
          </Badge>
        </Group>
        <Text size="xs" c="dimmed">
          {new Date(version.createdAt).toLocaleDateString()}
        </Text>
      </Group>

      <SimpleGrid cols={3} spacing="sm">
        {SPLIT_TYPES.map((splitType) => (
          <Card
            key={splitType}
            withBorder
            p="sm"
            radius="sm"
            className="card-elevated"
            style={{ cursor: 'pointer' }}
            onClick={() => onSelectSplit(version.id, splitType)}
          >
            <Group justify="space-between">
              <Text size="sm" fw={500} tt="capitalize">
                {splitType}
              </Text>
              <Badge size="xs" variant="light" color={SPLIT_COLORS[splitType] ?? 'gray'}>
                {counts[splitType]}
              </Badge>
            </Group>
          </Card>
        ))}
      </SimpleGrid>
    </Card>
  )
}

/* ── Main Dataset Page ── */
export function DatasetPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id
  const activeProject = useProjectStore((s) => s.activeProject)
  const dataset = activeProject?.dataset

  const [createVersionOpened, { open: openCreateVersion, close: closeCreateVersion }] = useDisclosure(false)
  const [selectedSplit, setSelectedSplit] = useState<{
    versionId: string
    type: 'train' | 'validation' | 'test'
  } | null>(null)

  return (
    <Box>
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={2}>Dataset</Title>
            <Text size="sm" c="dimmed" mt={4}>
              {activeProject ? `Manage dataset for "${activeProject.name}"` : 'Manage dataset'}
            </Text>
          </div>
          {dataset && (
            <Group gap="sm">
              <Badge color={MODALITY_COLORS[dataset.modality]} variant="light">
                {dataset.modality}
              </Badge>
              <Button size="xs" leftSection={<PlusIcon size={14} />} variant="light" onClick={openCreateVersion}>
                Create Snapshot
              </Button>
            </Group>
          )}
        </Group>

        {!dataset ? (
          <Card withBorder p="xl" radius="md" ta="center">
            <Stack align="center" gap="md">
              <ThemeIcon size={56} variant="light" color="gray" radius="xl">
                <DatabaseIcon size={30} weight="thin" />
              </ThemeIcon>
              <Title order={5}>No dataset found</Title>
              <Text size="sm" c="dimmed">
                This project's dataset hasn't been initialized yet.
              </Text>
            </Stack>
          </Card>
        ) : (
          <Stack gap="lg">
            {/* Draft version */}
            {dataset.draft && (
              <VersionCard
                version={dataset.draft}
                isDraft
                onSelectSplit={(versionId, type) => setSelectedSplit({ versionId, type })}
              />
            )}

            {/* Snapshot versions */}
            {(dataset.versions?.length ?? 0) > 0 && (
              <div>
                <Title order={5} mb="md">
                  Snapshots
                </Title>
                <Stack gap="sm">
                  {dataset.versions.map((version) => (
                    <VersionCard
                      key={version.id}
                      version={version}
                      isDraft={false}
                      onSelectSplit={(versionId, type) => setSelectedSplit({ versionId, type })}
                    />
                  ))}
                </Stack>
              </div>
            )}

            {(dataset.versions?.length ?? 0) === 0 && (
              <Card withBorder p="md" radius="md">
                <Text size="sm" c="dimmed" ta="center">
                  No snapshots yet. Create one to freeze the current dataset state for training.
                </Text>
              </Card>
            )}

            {/* Selected split items */}
            {selectedSplit && (
              <Card withBorder p="lg" radius="md">
                <Group justify="space-between" mb="md">
                  <Title order={5}>Items</Title>
                  <Button size="xs" variant="subtle" onClick={() => setSelectedSplit(null)}>
                    Close
                  </Button>
                </Group>
                <SplitItemsPanel
                  projectId={projectId}
                  versionId={selectedSplit.versionId}
                  splitType={selectedSplit.type}
                />
              </Card>
            )}
          </Stack>
        )}
      </Stack>

      {/* Create Version Modal */}
      <Modal opened={createVersionOpened} onClose={closeCreateVersion} title="Create Snapshot Version" centered>
        <CreateVersionModal projectId={projectId} onClose={closeCreateVersion} />
      </Modal>
    </Box>
  )
}

import {
  Button,
  Chip,
  Group,
  Paper,
  SegmentedControl,
  Select,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  TextInput,
} from '@mantine/core'
import { FoldersIcon, ListIcon, MagnifyingGlassIcon, PlusIcon, SquaresFourIcon } from '@phosphor-icons/react'
import { openCreateProjectModal } from '@public/components/CreateProjectModal'
import { ProjectCard, ProjectRow } from '@public/components/projects/ProjectItem'
import { EmptyState, PageHeader, QueryBoundary, SectionLabel } from '@public/components/ui'
import type { ProjectSummary } from '@public/lib/api/generated/models'
import { MODALITY_META } from '@public/lib/modality'
import { filterProjects, type ProjectSort } from '@public/lib/projectsList'
import { useProjects } from '@public/lib/queries'
import { taskRegistry } from '@public/lib/tasks'
import { useMemo, useState } from 'react'

type View = 'list' | 'cards'
const VIEW_KEY = 'omnitrain.projects.view'

const SORTS: { value: ProjectSort; label: string }[] = [
  { value: 'recent', label: 'Recent activity' },
  { value: 'name', label: 'Name' },
  { value: 'created', label: 'Date created' },
]

const modalityOf = (task: string): string => taskRegistry[task as keyof typeof taskRegistry]?.modality ?? 'vision'

/** The layout is a per-person preference; storage can be unavailable (private window), so it only ever falls back. */
function readView(): View {
  try {
    return localStorage.getItem(VIEW_KEY) === 'cards' ? 'cards' : 'list'
  } catch {
    return 'list'
  }
}

function Items({ projects, view }: { projects: ProjectSummary[]; view: View }) {
  return view === 'cards' ? (
    <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }} spacing="sm">
      {projects.map((p) => (
        <ProjectCard key={p.id} p={p} />
      ))}
    </SimpleGrid>
  ) : (
    <Stack gap="xs">
      {projects.map((p) => (
        <ProjectRow key={p.id} p={p} />
      ))}
    </Stack>
  )
}

export function DashboardPage() {
  const { data, isLoading, isError, refetch } = useProjects()
  const projects = useMemo(() => data?.projects ?? [], [data?.projects])

  const [q, setQ] = useState('')
  const [modality, setModality] = useState<string | null>(null)
  const [sort, setSort] = useState<ProjectSort>('recent')
  const [view, setView] = useState<View>(readView)
  const changeView = (next: View) => {
    setView(next)
    try {
      localStorage.setItem(VIEW_KEY, next)
    } catch {
      // not remembered; the choice still applies for this visit
    }
  }

  // Only the modalities the person actually has projects in are worth a filter.
  const modalities = useMemo(
    () => [...new Set(projects.map((p) => modalityOf(p.task)))].filter((m) => m in MODALITY_META),
    [projects],
  )
  const visible = useMemo(
    () => filterProjects(projects, { q, modality, sort }, modalityOf),
    [projects, q, modality, sort],
  )
  const inProgress = visible.filter((p) => p.activeRun)
  const rest = visible.filter((p) => !p.activeRun)
  const filtering = q.trim() !== '' || modality !== null

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Projects"
        description={projects.length > 0 ? 'Pick up where you left off.' : undefined}
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
          <Stack gap="xs">
            {Array.from({ length: 5 }, (_, i) => `skeleton-${i}`).map((key) => (
              <Paper key={key} p="sm">
                <Group wrap="nowrap" gap="md">
                  <Skeleton height={40} width={40} radius="md" />
                  <Stack gap={6} style={{ flex: 1 }}>
                    <Skeleton height={14} width="30%" radius="sm" />
                    <Skeleton height={11} width="60%" radius="sm" />
                  </Stack>
                  <Skeleton height={26} width={96} radius="sm" />
                </Group>
              </Paper>
            ))}
          </Stack>
        }
      >
        {projects.length === 0 ? (
          <EmptyState
            icon={FoldersIcon}
            title="Create your first project"
            description="A project holds one dataset and the models you train on it. Pick what you want to teach a model and start adding data."
            action={
              <Button leftSection={<PlusIcon size={15} />} onClick={openCreateProjectModal}>
                New project
              </Button>
            }
          />
        ) : (
          <>
            <Group gap="sm" wrap="wrap" align="center">
              <TextInput
                size="sm"
                placeholder="Search projects…"
                aria-label="Search projects"
                leftSection={<MagnifyingGlassIcon size={16} />}
                value={q}
                onChange={(e) => setQ(e.currentTarget.value)}
                style={{ flexGrow: 1, minWidth: 200, maxWidth: 320 }}
              />
              {modalities.length > 1 && (
                <Chip.Group
                  multiple={false}
                  value={modality ?? 'all'}
                  onChange={(v) => setModality(v === 'all' ? null : v)}
                >
                  <Group gap={6}>
                    <Chip value="all" size="sm" variant="light">
                      All
                    </Chip>
                    {modalities.map((m) => (
                      <Chip
                        key={m}
                        value={m}
                        size="sm"
                        variant="light"
                        color={MODALITY_META[m as keyof typeof MODALITY_META].color}
                      >
                        {MODALITY_META[m as keyof typeof MODALITY_META].label}
                      </Chip>
                    ))}
                  </Group>
                </Chip.Group>
              )}
              <Group gap="xs" ml="auto" wrap="nowrap">
                <Select
                  size="sm"
                  w={170}
                  aria-label="Sort projects"
                  allowDeselect={false}
                  data={SORTS}
                  value={sort}
                  onChange={(v) => v && setSort(v as ProjectSort)}
                />
                <SegmentedControl
                  size="md"
                  value={view}
                  onChange={(v) => changeView(v as View)}
                  data={[
                    { value: 'list', label: <ListIcon size={16} aria-label="List view" /> },
                    { value: 'cards', label: <SquaresFourIcon size={16} aria-label="Card view" /> },
                  ]}
                />
              </Group>
            </Group>

            {visible.length === 0 ? (
              <EmptyState
                compact
                title="No projects match"
                description={filtering ? 'Try a different search, or clear the filters.' : undefined}
                action={
                  filtering ? (
                    <Button
                      variant="default"
                      onClick={() => {
                        setQ('')
                        setModality(null)
                      }}
                    >
                      Clear filters
                    </Button>
                  ) : undefined
                }
              />
            ) : (
              <Stack gap="md">
                {inProgress.length > 0 && (
                  <Stack gap="xs">
                    <SectionLabel>In progress</SectionLabel>
                    <Items projects={inProgress} view={view} />
                  </Stack>
                )}
                {rest.length > 0 && (
                  <Stack gap="xs">
                    {inProgress.length > 0 && <SectionLabel>Other projects</SectionLabel>}
                    <Items projects={rest} view={view} />
                  </Stack>
                )}
              </Stack>
            )}
            {filtering && visible.length > 0 && (
              <Text size="xs" c="dimmed">
                Showing {visible.length} of {projects.length} projects.
              </Text>
            )}
          </>
        )}
      </QueryBoundary>
    </div>
  )
}

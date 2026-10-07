import {
  ActionIcon,
  Avatar,
  Badge,
  Box,
  Group,
  Image,
  AppShell as MantineAppShell,
  Menu,
  NavLink,
  ScrollArea,
  Select,
  Stack,
  Text,
  UnstyledButton,
  useMantineColorScheme,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import {
  ArchiveIcon,
  BrainIcon,
  CaretDownIcon,
  CubeIcon,
  DatabaseIcon,
  ExportIcon,
  FoldersIcon,
  GearIcon,
  ListIcon,
  MoonIcon,
  PlayIcon,
  ShieldCheckIcon,
  SignOutIcon,
  StackIcon,
  SunIcon,
  TagIcon,
  UploadSimpleIcon,
} from '@phosphor-icons/react'
import Logo from '@public/assets/CTU_logo.png'
import { SectionLabel } from '@public/components/ui'
import { logout } from '@public/lib/api/generated/auth/auth'
import { sessionQueryOptions } from '@public/lib/auth'
import { projectDetailQueryOptions, useProjects } from '@public/lib/queries'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useMatch, useNavigate } from '@tanstack/react-router'
import { Fragment, type ReactNode } from 'react'

const hasReadyVersion = (project: ProjectDetail) =>
  project.dataset?.versions?.some((v) => v.status === 'ready') ?? false

interface ProjectNavItem {
  label: string
  icon: typeof DatabaseIcon
  to:
    | '/project/$projectId'
    | '/project/$projectId/upload'
    | '/project/$projectId/classes'
    | '/project/$projectId/dataset'
    | '/project/$projectId/snapshots'
    | '/project/$projectId/training'
    | '/project/$projectId/playground'
    | '/project/$projectId/export'
  /** Overview must match exactly; everything else matches by prefix. */
  exact?: boolean
  /** A heading shown above the first visible item of each group, so the data steps read as one flow. */
  group?: string
  color?: string
  hidden?: (project: ProjectDetail) => boolean
  disabled?: (project: ProjectDetail) => boolean
  count?: (project: ProjectDetail) => number
}

/** Filled badges carry light text, so use a shade dark enough for 4.5:1 (the light variant keeps the plain color). */
function filledBadgeColor(color: string | undefined) {
  if (!color) return 'gray.7'
  return color === 'primary' ? color : `${color}.${color === 'teal' ? 9 : 8}`
}

const PROJECT_NAV_ITEMS: ProjectNavItem[] = [
  { label: 'Overview', icon: DatabaseIcon, to: '/project/$projectId', exact: true },
  { label: 'Upload', icon: UploadSimpleIcon, to: '/project/$projectId/upload', color: 'primary', group: 'Data' },
  {
    label: 'Classes',
    icon: TagIcon,
    to: '/project/$projectId/classes',
    /** Only visible for tasks that use label classes */
    hidden: (project) => !isClassificationTask(project.task),
    count: (project) => project.dataset?.classes?.length ?? 0,
    color: 'violet',
    group: 'Data',
  },
  {
    label: 'Dataset',
    icon: StackIcon,
    to: '/project/$projectId/dataset',
    count: (project) => project.dataset?.draft?.itemCount ?? 0,
    color: 'blue',
    group: 'Data',
  },
  {
    label: 'Snapshots',
    icon: ArchiveIcon,
    to: '/project/$projectId/snapshots',
    count: (project) => project.dataset?.versions?.length ?? project.versionCount ?? 0,
    color: 'grape',
    group: 'Data',
  },
  {
    label: 'Training',
    icon: BrainIcon,
    to: '/project/$projectId/training',
    disabled: (project) => !hasReadyVersion(project),
    count: (project) => project.runCount ?? 0,
    color: 'teal',
    group: 'Models',
  },
  {
    label: 'Playground',
    icon: PlayIcon,
    to: '/project/$projectId/playground',
    disabled: (project) => (project.runCount ?? 0) === 0,
    group: 'Models',
  },
  {
    label: 'Export',
    icon: ExportIcon,
    to: '/project/$projectId/export',
    disabled: (project) => (project.runCount ?? 0) === 0,
    group: 'Models',
  },
]

function ProjectSwitcher({ projectId }: { projectId: string | undefined }) {
  const navigate = useNavigate()
  const { data } = useProjects()
  const projects = data?.projects ?? []

  return (
    <Group gap="xs" wrap="nowrap">
      <Select
        size="xs"
        w={220}
        searchable
        placeholder="Select project"
        aria-label="Switch project"
        leftSection={<FoldersIcon size={14} />}
        data={projects.map((p) => ({ value: p.id, label: p.name }))}
        value={projectId ?? null}
        onChange={(id) => id && navigate({ to: '/project/$projectId', params: { projectId: id } })}
        nothingFoundMessage="No projects"
      />
    </Group>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [mobileOpened, { toggle: toggleMobile }] = useDisclosure()
  const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true)
  const { data: user } = useQuery(sessionQueryOptions)
  const { colorScheme, setColorScheme } = useMantineColorScheme()
  const isAdmin = user?.role === 'admin'
  const toggleColorScheme = () => setColorScheme(colorScheme === 'dark' ? 'light' : 'dark')

  // AppShell sits above the project route, so it can't read that route's loader — this
  // non-throwing match is undefined outside /project/$projectId.
  const projectMatch = useMatch({ from: '/_app/project/$projectId', shouldThrow: false })
  const projectId = projectMatch?.params.projectId
  const { data: projectData } = useQuery({
    ...projectDetailQueryOptions(projectId ?? ''),
    enabled: !!projectId,
  })
  const activeProject = projectData?.project
  const visibleNavItems = activeProject ? PROJECT_NAV_ITEMS.filter((item) => !item.hidden?.(activeProject)) : []

  const handleLogout = async () => {
    await logout().catch(() => undefined) // the cookies are cleared server-side; a network failure must not trap the user
    queryClient.clear() // drop the previous user's cached data
    queryClient.setQueryData(sessionQueryOptions.queryKey, null)
    navigate({ to: '/login' })
  }

  const initials = (user?.name ?? 'U')
    .split(/\s+/)
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  return (
    <MantineAppShell
      header={{ height: 48 }}
      navbar={{
        width: 208,
        breakpoint: 'sm',
        collapsed: { mobile: !mobileOpened, desktop: !desktopOpened },
      }}
      padding={0}
    >
      {/* ─── Header ─── */}
      <MantineAppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="md" wrap="nowrap">
            <ActionIcon variant="subtle" color="gray" hiddenFrom="sm" onClick={toggleMobile} aria-label="Toggle menu">
              <ListIcon size={18} />
            </ActionIcon>
            <ActionIcon
              variant="subtle"
              color="gray"
              visibleFrom="sm"
              onClick={toggleDesktop}
              aria-label="Toggle sidebar"
            >
              <ListIcon size={18} />
            </ActionIcon>
            <UnstyledButton onClick={() => navigate({ to: '/projects' })} aria-label="CTU OmniTrain home">
              <Group gap={8} wrap="nowrap">
                <Image src={Logo} alt="" w={28} h={28} fit="contain" flex="none" />
                <Text fw={600} size="md" style={{ letterSpacing: 0.3 }}>
                  CTU OmniTrain
                </Text>
              </Group>
            </UnstyledButton>
            <ProjectSwitcher projectId={projectId} />
          </Group>

          <Group gap="xs" wrap="nowrap">
            <ActionIcon variant="subtle" color="gray" onClick={toggleColorScheme} aria-label="Toggle color scheme">
              {colorScheme === 'dark' ? <SunIcon size={16} /> : <MoonIcon size={16} />}
            </ActionIcon>

            <Menu shadow="md" width={200} position="bottom-end">
              <Menu.Target>
                <UnstyledButton>
                  <Group gap="xs" wrap="nowrap">
                    <Avatar size="sm" radius="xl" color="cyan">
                      {initials}
                    </Avatar>
                    <Text size="sm" fw={500} visibleFrom="sm">
                      {user?.name ?? 'User'}
                    </Text>
                    <CaretDownIcon size={12} />
                  </Group>
                </UnstyledButton>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Item leftSection={<FoldersIcon size={14} />} onClick={() => navigate({ to: '/projects' })}>
                  All projects
                </Menu.Item>
                <Menu.Item leftSection={<CubeIcon size={14} />} onClick={() => navigate({ to: '/imported-models' })}>
                  Imported models
                </Menu.Item>
                <Menu.Item leftSection={<GearIcon size={14} />} onClick={() => navigate({ to: '/settings' })}>
                  Settings &amp; API keys
                </Menu.Item>
                {isAdmin && (
                  <Menu.Item leftSection={<ShieldCheckIcon size={14} />} onClick={() => navigate({ to: '/admin' })}>
                    Administration
                  </Menu.Item>
                )}
                <Menu.Divider />
                <Menu.Item leftSection={<SignOutIcon size={14} />} color="red" onClick={handleLogout}>
                  Sign out
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Group>
      </MantineAppShell.Header>

      {/* ─── Navbar ─── */}
      <MantineAppShell.Navbar p="xs">
        <MantineAppShell.Section grow component={ScrollArea} scrollbarSize={4}>
          <Stack gap={2}>
            <NavLink
              label="All projects"
              leftSection={<FoldersIcon size={18} />}
              active={location.pathname === '/projects' || location.pathname === '/projects/'}
              onClick={() => navigate({ to: '/projects' })}
              variant="light"
            />
            <NavLink
              label="Imported models"
              leftSection={<CubeIcon size={18} />}
              active={location.pathname === '/imported-models' || location.pathname.startsWith('/imported-models/')}
              onClick={() => navigate({ to: '/imported-models' })}
              variant="light"
            />
            {isAdmin && (
              <NavLink
                label="Administration"
                leftSection={<ShieldCheckIcon size={18} />}
                active={location.pathname === '/admin' || location.pathname.startsWith('/admin/')}
                onClick={() => navigate({ to: '/admin' })}
                variant="light"
              />
            )}
            {activeProject && (
              <>
                <SectionLabel px="xs" pt="sm" pb={2} truncate>
                  {activeProject.name}
                </SectionLabel>
                {visibleNavItems.map((item, index) => {
                  const resolvedPath = item.to.replace('$projectId', activeProject.id)
                  const count = item.count?.(activeProject)
                  const isActive = item.exact
                    ? location.pathname === resolvedPath || location.pathname === `${resolvedPath}/`
                    : location.pathname === resolvedPath || location.pathname.startsWith(`${resolvedPath}/`)

                  return (
                    <Fragment key={item.to}>
                      {item.group && item.group !== visibleNavItems[index - 1]?.group && (
                        <SectionLabel px="xs" pt="xs" pb={2}>
                          {item.group}
                        </SectionLabel>
                      )}
                      <NavLink
                        label={item.label}
                        leftSection={<item.icon size={18} />}
                        rightSection={
                          count !== undefined ? (
                            <Badge
                              size="xs"
                              variant={isActive ? 'filled' : 'light'}
                              color={isActive ? filledBadgeColor(item.color) : (item.color ?? 'gray')}
                            >
                              {count}
                            </Badge>
                          ) : null
                        }
                        active={isActive}
                        onClick={() => navigate({ to: item.to, params: { projectId: activeProject.id } })}
                        variant="light"
                        disabled={item.disabled?.(activeProject)}
                      />
                    </Fragment>
                  )
                })}
              </>
            )}
          </Stack>
        </MantineAppShell.Section>

        <MantineAppShell.Section>
          <Box px="xs" py={4}>
            <Text size="xs" c="dimmed">
              CTU OmniTrain v1.5
            </Text>
          </Box>
        </MantineAppShell.Section>
      </MantineAppShell.Navbar>

      {/* ─── Main ─── */}
      <MantineAppShell.Main>{children}</MantineAppShell.Main>
    </MantineAppShell>
  )
}

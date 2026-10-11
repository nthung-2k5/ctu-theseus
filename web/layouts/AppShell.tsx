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
  FoldersIcon,
  GearIcon,
  ListIcon,
  MoonIcon,
  SignOutIcon,
  SquaresFourIcon,
  SunIcon,
} from '@phosphor-icons/react'
import Logo from '@public/assets/CTU_logo.png'
import { SectionLabel } from '@public/components/ui'
import { logout } from '@public/lib/api/generated/auth/auth'
import { isAdmin, sessionQueryOptions } from '@public/lib/auth'
import { projectDetailQueryOptions, useProjects } from '@public/lib/queries'
import type { ProjectDetail } from '@public/store/types'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useMatch, useNavigate } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { ADMIN_NAV_ITEMS, isAdminNavActive } from './adminNav'

const hasReadyVersion = (project: ProjectDetail) =>
  project.dataset?.versions?.some((v) => v.status === 'ready') ?? false

interface ProjectNavItem {
  label: string
  icon: typeof DatabaseIcon
  to:
    | '/project/$projectId'
    | '/project/$projectId/data'
    | '/project/$projectId/snapshots'
    | '/project/$projectId/training'
  /** Overview must match exactly; everything else matches by prefix. */
  exact?: boolean
  disabled?: (project: ProjectDetail) => boolean
  count?: (project: ProjectDetail) => number
}

/**
 * The project's flow, top to bottom. Counts are plain grey on purpose: colour in this app means status or a
 * dataset split, so a coloured count would read as one of those.
 */
const PROJECT_NAV_ITEMS: ProjectNavItem[] = [
  { label: 'Overview', icon: SquaresFourIcon, to: '/project/$projectId', exact: true },
  {
    label: 'Data',
    icon: DatabaseIcon,
    to: '/project/$projectId/data',
    count: (project) => project.dataset?.draft?.itemCount ?? 0,
  },
  {
    label: 'Snapshots',
    icon: ArchiveIcon,
    to: '/project/$projectId/snapshots',
    count: (project) => project.dataset?.versions?.length ?? project.versionCount ?? 0,
  },
  {
    label: 'Training',
    icon: BrainIcon,
    to: '/project/$projectId/training',
    disabled: (project) => !hasReadyVersion(project),
    count: (project) => project.runCount ?? 0,
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
  // An admin's shell is the admin area only: no project switcher, project nav, models or settings.
  const adminOnly = isAdmin(user ?? null)
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
            <UnstyledButton
              onClick={() => navigate({ to: adminOnly ? '/admin' : '/projects' })}
              aria-label="CTU OmniTrain home"
            >
              <Group gap={8} wrap="nowrap">
                <Image src={Logo} alt="" w={28} h={28} fit="contain" flex="none" />
                <Text fw={600} size="md" style={{ letterSpacing: 0.3 }}>
                  CTU OmniTrain
                </Text>
              </Group>
            </UnstyledButton>
            {!adminOnly && <ProjectSwitcher projectId={projectId} />}
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
                {!adminOnly && (
                  <>
                    <Menu.Item leftSection={<GearIcon size={14} />} onClick={() => navigate({ to: '/settings' })}>
                      Settings
                    </Menu.Item>
                    <Menu.Divider />
                  </>
                )}
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
            {adminOnly ? (
              <>
                <SectionLabel px="xs" pb={2}>
                  Administration
                </SectionLabel>
                {ADMIN_NAV_ITEMS.map((item) => (
                  <NavLink
                    key={item.to}
                    label={item.label}
                    leftSection={<item.icon size={18} />}
                    active={isAdminNavActive(item, location.pathname)}
                    onClick={() => navigate({ to: item.to })}
                    variant="light"
                  />
                ))}
              </>
            ) : (
              <>
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
              </>
            )}
            {activeProject && (
              <>
                <SectionLabel px="xs" pt="sm" pb={2} truncate>
                  {activeProject.name}
                </SectionLabel>
                {PROJECT_NAV_ITEMS.map((item) => {
                  const resolvedPath = item.to.replace('$projectId', activeProject.id)
                  const count = item.count?.(activeProject)
                  const isActive = item.exact
                    ? location.pathname === resolvedPath || location.pathname === `${resolvedPath}/`
                    : location.pathname === resolvedPath || location.pathname.startsWith(`${resolvedPath}/`)

                  return (
                    <NavLink
                      key={item.to}
                      label={item.label}
                      leftSection={<item.icon size={18} />}
                      rightSection={
                        count !== undefined ? (
                          <Badge size="xs" variant="light" color="gray">
                            {count}
                          </Badge>
                        ) : null
                      }
                      active={isActive}
                      onClick={() => navigate({ to: item.to, params: { projectId: activeProject.id } })}
                      variant="light"
                      disabled={item.disabled?.(activeProject)}
                    />
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

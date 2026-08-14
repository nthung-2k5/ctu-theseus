import {
  ActionIcon,
  Avatar,
  Box,
  Divider,
  Group,
  AppShell as MantineAppShell,
  Menu,
  NavLink,
  ScrollArea,
  Stack,
  Text,
  Title,
  UnstyledButton,
  useMantineColorScheme,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import {
  ArrowLeftIcon,
  BrainIcon,
  CaretDownIcon,
  CrosshairIcon,
  DatabaseIcon,
  HouseIcon,
  ListIcon,
  MoonIcon,
  PackageIcon,
  SignOutIcon,
  StackIcon,
  SunIcon,
  TagIcon,
  UploadSimpleIcon,
} from '@phosphor-icons/react'
import { authClient, sessionQueryOptions } from '@public/lib/auth'
import { projectDetailQueryOptions } from '@public/lib/queries'
import type { ProjectDetail } from '@public/store/types'
import { isClassificationTask } from '@server/lib/tasks'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useMatch, useNavigate } from '@tanstack/react-router'
import type { ReactNode } from 'react'

interface AppShellProps {
  children: ReactNode
}

const NAV_ITEMS = [{ label: 'Dashboard', icon: HouseIcon, to: '/' as const }]

const hasReadyVersion = (project: ProjectDetail) =>
  project.dataset?.versions?.some((v) => v.status === 'ready') ?? false

const PROJECT_NAV_ITEMS = [
  { label: 'Overview', icon: DatabaseIcon, to: '/project/$projectId' as const },
  { label: 'Data', icon: UploadSimpleIcon, to: '/project/$projectId/data' as const },
  { label: 'Dataset', icon: StackIcon, to: '/project/$projectId/dataset' as const },
  {
    label: 'Classes',
    icon: TagIcon,
    to: '/project/$projectId/classes' as const,
    /** Only visible for tasks that use label classes */
    hidden: (project: ProjectDetail) => !isClassificationTask(project.task),
  },
  {
    label: 'Training',
    icon: BrainIcon,
    to: '/project/$projectId/training' as const,
    disabled: (project: ProjectDetail) => !hasReadyVersion(project),
  },
  {
    label: 'Models',
    icon: PackageIcon,
    to: '/project/$projectId/models' as const,
    disabled: (project: ProjectDetail) => (project.runCount ?? 0) === 0,
  },
  {
    label: 'Inference',
    icon: CrosshairIcon,
    to: '/project/$projectId/inference' as const,
    disabled: (project: ProjectDetail) => (project.runCount ?? 0) === 0,
  },
]

export function AppShell({ children }: AppShellProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [mobileOpened, { toggle: toggleMobile }] = useDisclosure()
  const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true)
  const { data: session } = authClient.useSession()
  const { colorScheme, setColorScheme } = useMantineColorScheme()
  const toggleColorScheme = () => setColorScheme(colorScheme === 'dark' ? 'light' : 'dark')

  // AppShell sits above the project route, so it can't read that route's
  // loader — this non-throwing match is undefined on the dashboard and
  // defined everywhere under /project/$projectId.
  const projectMatch = useMatch({ from: '/_app/project/$projectId', shouldThrow: false })
  const projectId = projectMatch?.params.projectId
  const { data: projectData } = useQuery({
    ...projectDetailQueryOptions(projectId ?? ''),
    enabled: !!projectId,
  })
  const activeProject = projectData?.project

  const handleLogout = async () => {
    await authClient.signOut()
    queryClient.invalidateQueries({ queryKey: sessionQueryOptions.queryKey })
    navigate({ to: '/login' })
  }

  return (
    <MantineAppShell
      header={{ height: 56 }}
      navbar={{
        width: 260,
        breakpoint: 'sm',
        collapsed: { mobile: !mobileOpened, desktop: !desktopOpened },
      }}
      padding="lg"
    >
      {/* ─── Header ─── */}
      <MantineAppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <ActionIcon variant="subtle" color="gray" hiddenFrom="sm" onClick={toggleMobile}>
              <ListIcon size={20} />
            </ActionIcon>
            <ActionIcon variant="subtle" color="gray" visibleFrom="sm" onClick={toggleDesktop}>
              <ListIcon size={20} />
            </ActionIcon>
            <UnstyledButton onClick={() => navigate({ to: '/' })}>
              <Title order={4} c="primary">
                CTU Theseus
              </Title>
            </UnstyledButton>
            {activeProject && (
              <>
                <Text size="sm" c="dimmed">
                  /
                </Text>
                <Text size="sm" fw={500}>
                  {activeProject.name}
                </Text>
              </>
            )}
          </Group>

          <Group gap="sm">
            <ActionIcon variant="subtle" color="gray" onClick={toggleColorScheme} aria-label="Toggle color scheme">
              {colorScheme === 'dark' ? <SunIcon size={18} /> : <MoonIcon size={18} />}
            </ActionIcon>

            <Menu shadow="md" width={200} position="bottom-end">
              <Menu.Target>
                <UnstyledButton>
                  <Group gap="xs">
                    <Avatar size="sm" radius="xl" color="primary">
                      {session?.user?.name?.[0]?.toUpperCase() ?? 'U'}
                    </Avatar>
                    <Text size="sm" fw={500} visibleFrom="sm">
                      {session?.user?.name ?? 'User'}
                    </Text>
                    <CaretDownIcon size={14} />
                  </Group>
                </UnstyledButton>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Item leftSection={<SignOutIcon size={16} />} color="red" onClick={handleLogout}>
                  Sign out
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Group>
      </MantineAppShell.Header>

      {/* ─── Navbar ─── */}
      <MantineAppShell.Navbar p="sm">
        <MantineAppShell.Section grow component={ScrollArea} scrollbarSize={4}>
          <Stack gap={4}>
            {!activeProject ? (
              NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  label={item.label}
                  leftSection={<item.icon size={20} />}
                  active={location.pathname === item.to}
                  onClick={() => navigate({ to: item.to })}
                  variant="light"
                />
              ))
            ) : (
              <>
                <NavLink
                  label="Back to Dashboard"
                  leftSection={<ArrowLeftIcon size={20} />}
                  onClick={() => navigate({ to: '/' })}
                  variant="light"
                />
                <Divider my="xs" label="Project" labelPosition="left" />
                {PROJECT_NAV_ITEMS.filter((item) => !item.hidden?.(activeProject)).map((item) => {
                  const resolvedPath = item.to.replace('$projectId', activeProject.id)
                  return (
                    <NavLink
                      key={item.to}
                      label={item.label}
                      leftSection={<item.icon size={20} />}
                      active={location.pathname === resolvedPath}
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
          <Divider my="xs" />
          <Box px="xs" py={4}>
            <Text size="xs" c="dimmed">
              CTU Theseus v1.5
            </Text>
          </Box>
        </MantineAppShell.Section>
      </MantineAppShell.Navbar>

      {/* ─── Main ─── */}
      <MantineAppShell.Main>{children}</MantineAppShell.Main>
    </MantineAppShell>
  )
}

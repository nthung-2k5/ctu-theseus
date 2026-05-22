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
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import {
  BrainIcon,
  CaretDownIcon,
  CrosshairIcon,
  DatabaseIcon,
  GearIcon,
  HouseIcon,
  ListIcon,
  MagicWandIcon,
  SignOutIcon,
  TagIcon,
  UserIcon,
} from '@phosphor-icons/react'
import { authClient } from '@public/lib/auth'
import { useProjectStore } from '@public/store/useProjectStore'
import type { ReactNode } from 'react'
import { Link, useLocation } from 'wouter'

interface AppShellProps {
  children: ReactNode
}

const NAV_ITEMS = [{ label: 'Dashboard', icon: HouseIcon, path: '/' }]

const PROJECT_NAV_ITEMS = [
  { label: 'Overview', icon: DatabaseIcon, path: '' },
  { label: 'Dataset', icon: DatabaseIcon, path: '/dataset' },
  { label: 'Labeling', icon: TagIcon, path: '/labeling' },
  { label: 'Augmentation', icon: MagicWandIcon, path: '/augmentation' },
  { label: 'Training', icon: BrainIcon, path: '/training' },
  { label: 'Inference', icon: CrosshairIcon, path: '/inference' },
]

export function AppShell({ children }: AppShellProps) {
  const [location, setLocation] = useLocation()
  const [mobileOpened, { toggle: toggleMobile }] = useDisclosure()
  const [desktopOpened, { toggle: toggleDesktop }] = useDisclosure(true)
  const { data: session } = authClient.useSession()
  const activeProject = useProjectStore((s) => s.activeProject)

  const handleLogout = async () => {
    await authClient.signOut()
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
            <Link href='/'>
              <Title order={4} c="primary">
                CTU Theseus
              </Title>
            </Link>
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
              <Menu.Label>Account</Menu.Label>
              <Menu.Item leftSection={<UserIcon size={16} />}>Profile</Menu.Item>
              <Menu.Item leftSection={<GearIcon size={16} />}>Settings</Menu.Item>
              <Menu.Divider />
              <Menu.Item leftSection={<SignOutIcon size={16} />} color="red" onClick={handleLogout}>
                Sign out
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </Group>
      </MantineAppShell.Header>

      {/* ─── Navbar ─── */}
      <MantineAppShell.Navbar p="sm">
        <MantineAppShell.Section grow component={ScrollArea} scrollbarSize={4}>
          <Stack gap={4}>
            {!activeProject ? (
              NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.path}
                  label={item.label}
                  leftSection={<item.icon size={20} />}
                  active={location === item.path}
                  onClick={() => setLocation(item.path)}
                  variant="light"
                />
              ))
            ) : (
              <>
                <Divider my="xs" label="Project" labelPosition="left" />
                {PROJECT_NAV_ITEMS.map((item) => {
                  const fullPath = `/project/${activeProject.id}${item.path}`
                  return (
                    <NavLink
                      key={item.path}
                      label={item.label}
                      leftSection={<item.icon size={20} />}
                      active={location === fullPath}
                      onClick={() => setLocation(fullPath)}
                      variant="light"
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
              CTU Theseus v0.1
            </Text>
          </Box>
        </MantineAppShell.Section>
      </MantineAppShell.Navbar>

      {/* ─── Main ─── */}
      <MantineAppShell.Main>{children}</MantineAppShell.Main>
    </MantineAppShell>
  )
}

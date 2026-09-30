import { Tabs } from '@mantine/core'
import { PageHeader } from '@public/components/ui'
import { Outlet, useLocation, useNavigate } from '@tanstack/react-router'

type AdminTab = 'overview' | 'users' | 'models' | 'plugins' | 'jobs'

interface TabItem {
  value: AdminTab
  label: string
  to: '/admin' | '/admin/users' | '/admin/models' | '/admin/plugins' | '/admin/jobs'
}

const TABS: TabItem[] = [
  { value: 'overview', label: 'Overview', to: '/admin' },
  { value: 'users', label: 'Users', to: '/admin/users' },
  { value: 'models', label: 'Models', to: '/admin/models' },
  { value: 'plugins', label: 'Plugins', to: '/admin/plugins' },
  { value: 'jobs', label: 'Jobs', to: '/admin/jobs' },
]

/** The Overview tab is the index route, so it matches exactly; every other tab matches by prefix. */
const isActive = (tab: TabItem, pathname: string) =>
  tab.to === '/admin' ? pathname === '/admin' || pathname === '/admin/' : pathname.startsWith(tab.to)

/** Shared frame of every /admin page: the title and the section tabs. Access is enforced by the route's beforeLoad. */
export function AdminLayout() {
  const location = useLocation()
  const navigate = useNavigate()
  const active = TABS.find((t) => isActive(t, location.pathname))?.value ?? null

  return (
    <div className="flex flex-col gap-3 p-3" style={{ maxWidth: 1200 }}>
      <PageHeader title="Administration" description="Manage accounts and the platform. Visible to admins only." />
      <Tabs
        value={active}
        onChange={(value) => {
          const tab = TABS.find((t) => t.value === value)
          if (tab) navigate({ to: tab.to })
        }}
      >
        <Tabs.List>
          {TABS.map((t) => (
            <Tabs.Tab key={t.value} value={t.value}>
              {t.label}
            </Tabs.Tab>
          ))}
        </Tabs.List>
      </Tabs>
      <Outlet />
    </div>
  )
}

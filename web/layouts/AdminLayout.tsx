import { Tabs } from '@mantine/core'
import { PageHeader } from '@public/components/ui'
import { Outlet, useLocation, useNavigate } from '@tanstack/react-router'

type AdminTab = 'users'

const TABS: { value: AdminTab; label: string; to: '/admin/users' }[] = [
  { value: 'users', label: 'Users', to: '/admin/users' },
]

/** Shared frame of every /admin page: the title and the section tabs. Access is enforced by the route's beforeLoad. */
export function AdminLayout() {
  const location = useLocation()
  const navigate = useNavigate()
  const active = TABS.find((t) => location.pathname.startsWith(t.to))?.value ?? null

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

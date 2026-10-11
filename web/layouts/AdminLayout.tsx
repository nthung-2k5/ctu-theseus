import { PageHeader } from '@public/components/ui'
import { Outlet } from '@tanstack/react-router'

/** Shared frame of every /admin page: the title. The sections live in the sidebar; access is enforced by the route's beforeLoad. */
export function AdminLayout() {
  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader title="Administration" description="Manage accounts and the platform. Visible to admins only." />
      <Outlet />
    </div>
  )
}

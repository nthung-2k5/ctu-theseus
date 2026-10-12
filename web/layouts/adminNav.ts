import { CpuIcon, CubeIcon, PuzzlePieceIcon, QueueIcon, SquaresFourIcon, UsersIcon } from '@phosphor-icons/react'

export interface AdminNavItem {
  label: string
  icon: typeof CubeIcon
  to: '/admin' | '/admin/users' | '/admin/models' | '/admin/plugins' | '/admin/jobs' | '/admin/workers'
}

/** The sections of the admin area; rendered as the sidebar when an admin is signed in. */
export const ADMIN_NAV_ITEMS: AdminNavItem[] = [
  { label: 'Overview', icon: SquaresFourIcon, to: '/admin' },
  { label: 'Users', icon: UsersIcon, to: '/admin/users' },
  { label: 'Models', icon: CubeIcon, to: '/admin/models' },
  { label: 'Plugins', icon: PuzzlePieceIcon, to: '/admin/plugins' },
  { label: 'Jobs', icon: QueueIcon, to: '/admin/jobs' },
  { label: 'Workers', icon: CpuIcon, to: '/admin/workers' },
]

/** The Overview item is the index route, so it matches exactly; every other item matches by prefix. */
export const isAdminNavActive = (item: AdminNavItem, pathname: string) =>
  item.to === '/admin' ? pathname === '/admin' || pathname === '/admin/' : pathname.startsWith(item.to)

/**
 * Shared display formatters.
 *
 * `formatDate` was previously copy-pasted into DashboardPage and ProjectPage,
 * and the two copies had already diverged (`isNaN` vs `Number.isNaN`).
 */

/** A short date ("12 Mar 2026"), or an em dash for missing/unparseable input. */
export function formatDate(dateInput: string | Date | undefined | null): string {
  if (!dateInput) return '—'
  const date = new Date(dateInput)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

/** A short date and time, for timestamps where the hour matters (run start/finish). */
export function formatDateTime(dateInput: string | Date | undefined | null): string {
  if (!dateInput) return '—'
  const date = new Date(dateInput)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** A compact file size ("2.4 MB"), for upload previews. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

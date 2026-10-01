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

/** `freezeBackbone` -> "Freeze Backbone": the last-resort label for a camelCase key nothing else names. */
export function humanizeKey(key: string): string {
  const words = key.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ')
  return words.replace(/\b\w/g, (c) => c.toUpperCase())
}

/**
 * Hyperparameters whose stored number means "off" rather than a quantity: the value the trainer backend declares
 * as `ParamSpec.disabledValue`. Kept here too because the run summaries must also format runs whose backend spec
 * is not at hand (an old run, a backend that is no longer installed).
 */
const DISABLED_VALUES: Record<string, number> = { earlyStopPatience: -1 }

/** A run's hyperparameter as shown in the run overview, sweep detail and run comparison. */
export function formatHyperparamValue(key: string, value: unknown): string {
  if (typeof value === 'number' && DISABLED_VALUES[key] === value) return 'Disabled'
  if (Array.isArray(value)) return value.length > 0 ? value.join(', ') : 'None'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  return String(value)
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

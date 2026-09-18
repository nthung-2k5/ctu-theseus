import { Button } from '@mantine/core'
import { ArrowClockwiseIcon, WarningCircleIcon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'
import { EmptyState } from './EmptyState'

/**
 * Renders the loading / error states of a TanStack Query so pages don't have
 * to.
 *
 * Pages used to branch on `isLoading` only and then fall through to
 * `data?.things ?? []`, which meant a *failed* request rendered the page's
 * empty state — "No items in the draft yet", "create your first project" —
 * telling the user their data doesn't exist when the API is simply down.
 * Nothing in the app read `isError`.
 *
 * Route-level error components don't cover this: `router.tsx`'s
 * `defaultErrorComponent` only catches throws from `useSuspenseQuery`, and
 * most of these are plain `useQuery`.
 */
export function QueryBoundary({
  isLoading,
  isError,
  onRetry,
  compact = false,
  loadingFallback,
  children,
}: {
  isLoading: boolean
  isError: boolean
  onRetry?: () => void
  compact?: boolean
  /** Use a page-specific skeleton instead of the default spinner card. */
  loadingFallback?: ReactNode
  children?: ReactNode
}) {
  if (isError) {
    return (
      <EmptyState
        icon={WarningCircleIcon}
        title="Couldn't load this"
        description="The request failed. This is usually a connection problem rather than missing data."
        compact={compact}
        action={
          onRetry && (
            <Button variant="light" leftSection={<ArrowClockwiseIcon size={16} />} onClick={onRetry}>
              Try again
            </Button>
          )
        }
      />
    )
  }

  if (isLoading) return loadingFallback ?? <EmptyState loading compact={compact} />

  return <>{children}</>
}

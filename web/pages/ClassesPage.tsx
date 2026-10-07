/**
 * Classes page – the label-class registry for classification tasks.
 *
 * Edits (add, rename, recolour, describe, merge, remove) build up in a local draft and are saved in one batch
 * (PUT /projects/:id/classes), so a half-finished rename never reaches the server. Classes are listed in
 * creation order. Snapshots are immutable: changes only apply to new snapshots and runs. The editor and its
 * parts live in components/classes; this page loads the data and picks the right state to show.
 */

import { Alert } from '@mantine/core'
import { InfoIcon } from '@phosphor-icons/react'
import { ClassManager } from '@public/components/classes/ClassManager'
import { PageHeader, QueryBoundary } from '@public/components/ui'
import { invalidateProjectScope, projectDetailQueryOptions, useClassStats, useLabelClasses } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/classes')

export function ClassesPage() {
  const { projectId } = routeApi.useParams()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const descriptor = getTaskDescriptor(project.task)
  const usesClasses = descriptor.annotation.requiresLabelClasses
  const { data, isLoading, isError, refetch } = useLabelClasses(usesClasses ? projectId : undefined)
  const saved = data?.classes ?? []

  // Per-class counts (every class, empty ones included, split by train/validation/test). A cheap grouped query,
  // not the dataset health report this page used to read; the page is usable without it.
  const stats = useClassStats(projectId, usesClasses)

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Classes"
        description={`The labels this project's ${descriptor.label.toLowerCase()} model predicts.`}
      />

      {!usesClasses ? (
        <Alert icon={<InfoIcon size={16} />} color="blue" title="This task has no classes">
          {descriptor.label} predicts a value rather than a label from a fixed set, so there is nothing to manage here.
        </Alert>
      ) : (
        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {/* key: a saved change from the server resets the editable draft */}
          <ClassManager
            key={saved.map((c) => `${c.classId}:${c.name}:${c.description}:${c.uiColorHex}`).join('|')}
            projectId={projectId}
            saved={saved}
            supportsMerge={descriptor.annotation.type === 'classification'}
            stats={stats.data}
            statsFailed={stats.isError}
            onRetryStats={() => stats.refetch()}
            onSaved={() => invalidateProjectScope(queryClient, projectId)}
          />
        </QueryBoundary>
      )}
    </div>
  )
}

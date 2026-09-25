import { Button, SegmentedControl } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ArrowLeftIcon } from '@phosphor-icons/react'
import { CreateRunPanel, type RunPrefill } from '@public/components/training/CreateRunPanel'
import { CreateSweepPanel, type SweepStartConfig } from '@public/components/training/CreateSweepPanel'
import { PageHeader } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import type { TrainBody } from '@public/lib/api/generated/models'
import { getCreateSweepMutationOptions } from '@public/lib/api/generated/sweeps/sweeps'
import { getListRunsQueryKey, getStartTrainingMutationOptions } from '@public/lib/api/generated/training/training'
import { projectDetailQueryOptions, projectSweepsQueryOptions, useTrainingRunDetail } from '@public/lib/queries'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useMemo } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/experiments/new')

export function NewExperimentPage() {
  const { projectId } = routeApi.useParams()
  const { mode = 'run', from } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  // "New run from this setup": read the source run's hyperparameters and snapshot.
  const { data: source } = useTrainingRunDetail(from, !!from)
  const prefill = useMemo<RunPrefill | undefined>(() => {
    const run = source?.run
    if (!run) return undefined
    return {
      name: run.name,
      datasetVersionId: run.datasetVersion.id,
      hyperparameters: (run.hyperparameters ?? {}) as Record<string, unknown>,
    }
  }, [source])

  const goBack = () => navigate({ to: '/project/$projectId/experiments', params: { projectId } })

  const startTraining = useMutation({
    ...getStartTrainingMutationOptions(),
    onSuccess: ({ run }) => {
      queryClient.invalidateQueries({ queryKey: getListRunsQueryKey(projectId) })
      void navigate({ to: '/project/$projectId/experiments/$runId', params: { projectId, runId: run.id } })
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to start the run'), color: 'red' })
    },
  })

  const startSweep = useMutation({
    ...getCreateSweepMutationOptions(),
    onSuccess: ({ sweep }) => {
      queryClient.invalidateQueries({ queryKey: projectSweepsQueryOptions(projectId).queryKey })
      notifications.show({ title: 'Sweep started', message: `Dispatching trials for "${sweep.name}"…`, color: 'blue' })
      void navigate({
        to: '/project/$projectId/experiments/sweeps/$sweepId',
        params: { projectId, sweepId: sweep.id },
      })
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to start sweep'), color: 'red' })
    },
  })

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="New experiment"
        description={project.name}
        actions={
          <>
            <SegmentedControl
              value={mode}
              onChange={(v) => navigate({ search: (prev) => ({ ...prev, mode: v as 'run' | 'sweep' }), replace: true })}
              data={[
                { value: 'run', label: 'Run' },
                { value: 'sweep', label: 'Sweep' },
              ]}
            />
            <Button variant="default" leftSection={<ArrowLeftIcon size={14} />} onClick={goBack}>
              Back
            </Button>
          </>
        }
      />

      {mode === 'sweep' ? (
        <CreateSweepPanel
          project={project}
          loading={startSweep.isPending}
          onStartSweep={(config: SweepStartConfig) => startSweep.mutate({ projectId, data: config })}
        />
      ) : (
        <CreateRunPanel
          project={project}
          prefill={prefill}
          loading={startTraining.isPending}
          onStartTraining={(config) => startTraining.mutate({ projectId, data: config as TrainBody })}
        />
      )}
    </div>
  )
}

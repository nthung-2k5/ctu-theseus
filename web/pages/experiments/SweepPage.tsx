import { Button } from '@mantine/core'
import { ArrowLeftIcon } from '@phosphor-icons/react'
import { SweepDetailPanel } from '@public/components/training/SweepDetailPanel'
import { PageHeader } from '@public/components/ui'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/experiments/sweeps/$sweepId')

export function SweepPage() {
  const { projectId, sweepId } = routeApi.useParams()
  const navigate = routeApi.useNavigate()

  const back = () => navigate({ to: '/project/$projectId/experiments', params: { projectId } })

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Sweep"
        description="One training run per hyperparameter combination"
        actions={
          <Button variant="default" leftSection={<ArrowLeftIcon size={14} />} onClick={back}>
            Back to experiments
          </Button>
        }
      />
      <SweepDetailPanel
        sweepId={sweepId}
        onBack={back}
        onOpenRun={(runId) => navigate({ to: '/project/$projectId/experiments/$runId', params: { projectId, runId } })}
      />
    </div>
  )
}

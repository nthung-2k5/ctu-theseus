import { PlayIcon } from '@phosphor-icons/react'
import { useRunContext } from '@public/components/training/RunContext'
import { RunInferencePanel } from '@public/components/training/RunInferencePanel'
import { EmptyState } from '@public/components/ui'

/** Provide an input, run this run's model, and inspect the prediction. */
export function RunTryPage() {
  const { projectId, run, status } = useRunContext()

  if (status !== 'succeeded')
    return (
      <EmptyState
        icon={PlayIcon}
        title="Not ready to try yet"
        description="A model can be tried once its run has finished successfully."
        compact
      />
    )

  return <RunInferencePanel projectId={projectId} run={run} />
}

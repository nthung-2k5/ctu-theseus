import { ChartBarIcon } from '@phosphor-icons/react'
import { EvaluationPanel } from '@public/components/training/EvaluationPanel'
import { useRunContext } from '@public/components/training/RunContext'
import { EmptyState } from '@public/components/ui'

export function RunEvaluationPage() {
  const { projectId, project, run, status } = useRunContext()

  if (status !== 'succeeded') {
    return (
      <EmptyState
        icon={ChartBarIcon}
        title="Evaluation is not available yet"
        description="The confusion matrix and per-class metrics appear once the run succeeds."
      />
    )
  }

  return <EvaluationPanel run={run} projectId={projectId} modality={project.dataset?.modality} />
}

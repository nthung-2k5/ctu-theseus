import { Group } from '@mantine/core'
import { BrainIcon } from '@phosphor-icons/react'
import { useRunContext } from '@public/components/training/RunContext'
import { RunOverviewPanel } from '@public/components/training/RunOverviewPanel'
import { LinkButton } from '@public/components/ui'

/** Read-only summary of what the run was started with, plus a shortcut to start another from the same setup. */
export function RunConfigPage() {
  const { projectId, project, run } = useRunContext()

  return (
    <div className="flex flex-col gap-3">
      <Group justify="flex-end">
        <LinkButton
          to="/project/$projectId/experiments/new"
          params={{ projectId }}
          search={{ mode: 'run', from: run.id }}
          leftSection={<BrainIcon size={14} />}
        >
          New run from this setup
        </LinkButton>
      </Group>
      <RunOverviewPanel run={run} task={project.task} projectId={projectId} />
    </div>
  )
}

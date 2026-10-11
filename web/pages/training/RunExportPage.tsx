import { ActionIcon, Code, CopyButton, Paper, Stack, Tabs, Text, Tooltip } from '@mantine/core'
import { CheckIcon, CopyIcon, ExportIcon } from '@phosphor-icons/react'
import { useRunContext } from '@public/components/training/RunContext'
import { RunExportPanel } from '@public/components/training/RunExportPanel'
import { EmptyState, SectionLabel } from '@public/components/ui'
import { getInferenceInputSpec } from '@public/lib/tasks'
import { MONO_STACK } from '@public/theme'
import { predictSnippets } from './snippets'

function Snippet({ code }: { code: string }) {
  return (
    <div style={{ position: 'relative' }}>
      <Code block style={{ fontFamily: MONO_STACK, fontSize: 12 }}>
        {code}
      </Code>
      <CopyButton value={code} timeout={1500}>
        {({ copied, copy }) => (
          <Tooltip label={copied ? 'Copied' : 'Copy'}>
            <ActionIcon
              variant="subtle"
              color={copied ? 'teal' : 'gray'}
              onClick={copy}
              aria-label="Copy snippet"
              style={{ position: 'absolute', top: 6, right: 6 }}
            >
              {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
    </div>
  )
}

/** Package this run's model to download, or call it over the prediction API. */
export function RunExportPage() {
  const { project, run, status } = useRunContext()

  if (status !== 'succeeded')
    return (
      <EmptyState
        icon={ExportIcon}
        title="Not ready to export yet"
        description="A model can be packaged once its run has finished successfully."
        compact
      />
    )

  const snippets = predictSnippets(run.id, getInferenceInputSpec(project.task), window.location.origin)

  return (
    <div className="flex flex-col gap-3">
      <div>
        <SectionLabel mb={6}>Formats</SectionLabel>
        <RunExportPanel run={run} />
      </div>

      <Paper p="md">
        <Stack gap="xs">
          <SectionLabel>Use it: prediction API</SectionLabel>
          <Text size="xs" c="dimmed">
            Authenticate with an API key (create one under Settings) sent as a bearer token. The response is the same
            prediction the Try tab shows.
          </Text>
          <Tabs defaultValue="curl">
            <Tabs.List>
              <Tabs.Tab value="curl">curl</Tabs.Tab>
              <Tabs.Tab value="python">Python</Tabs.Tab>
            </Tabs.List>
            <Tabs.Panel value="curl" pt="xs">
              <Snippet code={snippets.curl} />
            </Tabs.Panel>
            <Tabs.Panel value="python" pt="xs">
              <Snippet code={snippets.python} />
            </Tabs.Panel>
          </Tabs>
        </Stack>
      </Paper>
    </div>
  )
}

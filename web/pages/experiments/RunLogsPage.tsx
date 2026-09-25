import { Badge, Button, Paper } from '@mantine/core'
import { DownloadSimpleIcon } from '@phosphor-icons/react'
import { LogConsole } from '@public/components/training/LogConsole'
import { useRunContext } from '@public/components/training/RunContext'
import { useRunLogs } from '@public/components/training/useRunLogs'

export function RunLogsPage() {
  const { run, isActive, live } = useRunContext()
  const { lines, isLoading } = useRunLogs()

  return (
    <Paper style={{ overflow: 'hidden' }}>
      <div style={{ height: 'calc(100vh - 260px)', minHeight: 320 }}>
        <LogConsole
          lines={lines}
          empty={
            isLoading ? 'Loading logs…' : isActive ? 'Waiting for output…' : 'No log output was recorded for this run.'
          }
          statusSlot={
            isActive ? (
              <Badge variant="dot" color={live.isConnected ? 'teal' : 'gray'}>
                {live.isConnected ? 'live' : 'connecting'}
              </Badge>
            ) : (
              <Button
                size="compact-xs"
                variant="light"
                color="gray"
                leftSection={<DownloadSimpleIcon size={12} />}
                component="a"
                href={`/api/runs/${run.id}/logs`}
                target="_blank"
                rel="noopener noreferrer"
              >
                Download
              </Button>
            )
          }
        />
      </div>
    </Paper>
  )
}

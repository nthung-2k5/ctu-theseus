import { Group, Paper, Text } from '@mantine/core'
import { PageHeader, SectionLabel } from '@public/components/ui'
import { sessionQueryOptions } from '@public/lib/auth'
import { ApiKeysSection } from '@public/pages/ApiKeysPage'
import { useQuery } from '@tanstack/react-query'

/** Account details plus the API keys used to call the hosted prediction API. */
export function SettingsPage() {
  const { data: user } = useQuery(sessionQueryOptions)

  return (
    <div className="flex flex-col gap-3 p-3" style={{ maxWidth: 980 }}>
      <PageHeader title="Settings" description="Your account and API access." />

      <Paper p="md">
        <SectionLabel mb="xs">Account</SectionLabel>
        <div className="flex flex-col gap-1">
          <Group justify="space-between">
            <Text size="xs" c="dimmed">
              Name
            </Text>
            <Text size="sm">{user?.name ?? '—'}</Text>
          </Group>
          <Group justify="space-between">
            <Text size="xs" c="dimmed">
              Email
            </Text>
            <Text size="sm">{user?.email ?? '—'}</Text>
          </Group>
        </div>
      </Paper>

      <ApiKeysSection />
    </div>
  )
}

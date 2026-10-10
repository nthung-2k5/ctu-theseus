import { Badge, Group, Paper, Text } from '@mantine/core'
import { HuggingFaceIntegration } from '@public/components/settings/HuggingFaceIntegration'
import { PageHeader } from '@public/components/ui'
import { sessionQueryOptions } from '@public/lib/auth'
import { ApiKeysSection } from '@public/pages/ApiKeysPage'
import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'

/** One titled block of the page, so what belongs together reads together. */
function SettingsSection({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <section className="flex flex-col gap-2" aria-label={title}>
      <div>
        <Text fw={600}>{title}</Text>
        <Text size="xs" c="dimmed">
          {description}
        </Text>
      </div>
      {children}
    </section>
  )
}

/** Your account, the outside services you have connected, and the keys that call the hosted prediction API. */
export function SettingsPage() {
  const { data: user } = useQuery(sessionQueryOptions)

  return (
    <div className="flex flex-col gap-5 p-3">
      <PageHeader title="Settings" description="Your account, connected services and API access." />

      <SettingsSection title="Account" description="How you are signed in.">
        <Paper p="md">
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
            <Group justify="space-between">
              <Text size="xs" c="dimmed">
                Role
              </Text>
              <Badge variant="light" color={user?.role === 'admin' ? 'grape' : 'gray'} tt="capitalize">
                {user?.role ?? '—'}
              </Badge>
            </Group>
          </div>
        </Paper>
      </SettingsSection>

      <SettingsSection
        title="Integrations"
        description="Connect outside services so Theseus can fetch things on your behalf."
      >
        <HuggingFaceIntegration />
      </SettingsSection>

      <SettingsSection title="API access" description="Keys for calling a trained model from your own code.">
        <ApiKeysSection />
      </SettingsSection>
    </div>
  )
}

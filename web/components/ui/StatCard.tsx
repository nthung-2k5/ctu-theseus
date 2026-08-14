import { Card, Group, Text, ThemeIcon } from '@mantine/core'
import type { Icon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'

/** A labeled stat tile with an icon — the "Dataset Versions / Training Runs / Modality" cards repeated across project pages. */
export function StatCard({
  icon: TheIcon,
  color = 'primary',
  label,
  value,
}: {
  icon: Icon
  color?: string
  label: string
  value: ReactNode
}) {
  return (
    <Card withBorder padding="lg" radius="md">
      <Group>
        <ThemeIcon size="lg" variant="light" color={color}>
          <TheIcon size={22} />
        </ThemeIcon>
        <div>
          <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
            {label}
          </Text>
          <Text size="xl" fw={700}>
            {value}
          </Text>
        </div>
      </Group>
    </Card>
  )
}

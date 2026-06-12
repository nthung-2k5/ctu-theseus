import { Checkbox, ColorSwatch, Group, Image, Paper, Stack, Text } from '@mantine/core'
import type { DatasetImage } from '@public/store/types'
import { useState } from 'react'
import { SplitBadgeIcon } from './SplitBadgeIcon'

interface ListRowProps {
  image: DatasetImage
  selected: boolean
  onSelect: (id: string, checked: boolean) => void
  cls?: { name: string; color: string }
}

export function ListRow({ image, selected, onSelect, cls }: ListRowProps) {
  const [hovered, setHovered] = useState(false)

  return (
    <Paper
      radius="sm"
      p="sm"
      withBorder
      style={{
        cursor: 'pointer',
        outline: selected ? '2px solid var(--mantine-primary-color-5)' : undefined,
        outlineOffset: -2,
        backgroundColor: hovered ? 'var(--mantine-color-dark-6)' : undefined,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onSelect(image.id, !selected)}
    >
      <Group gap="md" wrap="nowrap">
        {/* Thumbnail */}
        <Paper radius="xs" style={{ width: 80, height: 80, overflow: 'hidden', flexShrink: 0 }}>
          <Image src={image.url} alt={image.filename} fit="cover" w="100%" h="100%" />
        </Paper>

        {/* Info */}
        <Stack gap={6} flex={1} miw={0}>
          <Group gap="sm" wrap="nowrap">
            <Text size="sm" c="dimmed" w={56} style={{ flexShrink: 0 }}>
              Filename
            </Text>
            <Text size="sm" fw={500} truncate="end">
              {image.filename}
            </Text>
          </Group>
          <Group gap="sm" wrap="nowrap">
            <Text size="sm" c="dimmed" w={56} style={{ flexShrink: 0 }}>
              Class
            </Text>
            {cls ? (
              <Group gap={6} wrap="nowrap">
                <ColorSwatch color={cls.color} size={12} />
                <Text size="sm">{cls.name}</Text>
              </Group>
            ) : (
              <Text size="sm" c="dimmed">
                —
              </Text>
            )}
          </Group>
          <Group gap="sm" wrap="nowrap">
            <Text size="sm" c="dimmed" w={56} style={{ flexShrink: 0 }}>
              Split
            </Text>
            {image.split ? (
              <Group gap={6} wrap="nowrap">
                <SplitBadgeIcon split={image.split} forceExpand />
              </Group>
            ) : (
              <Text size="sm" c="dimmed">
                —
              </Text>
            )}
          </Group>
        </Stack>

        {/* Checkbox */}
        {(hovered || selected) && (
          <Checkbox
            checked={selected}
            onChange={(e) => {
              e.stopPropagation()
              onSelect(image.id, e.currentTarget.checked)
            }}
            onClick={(e) => e.stopPropagation()}
            size="sm"
            styles={{ input: { cursor: 'pointer' } }}
          />
        )}
      </Group>
    </Paper>
  )
}

import { Box, Text, Tooltip } from '@mantine/core'
import { BrainIcon, ShieldCheckIcon, TestTubeIcon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'
import { useState } from 'react'

const SPLIT_ICONS: Record<string, { icon: ReactNode; bg: string; fg: string; label: string }> = {
  train: {
    icon: <BrainIcon size={16} weight="bold" />,
    bg: 'var(--mantine-color-green-1)',
    fg: 'var(--mantine-color-green-7)',
    label: 'Train',
  },
  validation: {
    icon: <ShieldCheckIcon size={16} weight="bold" />,
    bg: 'var(--mantine-color-blue-1)',
    fg: 'var(--mantine-color-blue-7)',
    label: 'Validation',
  },
  test: {
    icon: <TestTubeIcon size={16} weight="bold" />,
    bg: 'var(--mantine-color-red-1)',
    fg: 'var(--mantine-color-red-7)',
    label: 'Test',
  },
}

export function SplitBadgeIcon({ split, forceExpand }: { split: string | null | undefined; forceExpand?: boolean }) {
  const [hovered, setHovered] = useState(false)
  if (!split || !SPLIT_ICONS[split]) return null
  const info = SPLIT_ICONS[split]
  const expanded = forceExpand || hovered

  const badge = (
    <Box
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      h={24}
      bdrs={5}
      bg={info.bg}
      c={info.fg}
      bd={`2px solid ${info.fg}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        maxWidth: expanded ? 120 : 24,
        paddingLeft: 4,
        paddingRight: expanded ? 6 : 4,
        transition: 'max-width 0.2s cubic-bezier(0.4, 0, 0.2, 1), padding 0.2s ease',
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        boxSizing: 'border-box',
      }}
    >
      <Box style={{ flexShrink: 0, display: 'flex', width: 14, justifyContent: 'center' }}>{info.icon}</Box>
      <Text
        size="xs"
        fw={600}
        ml={4}
        style={{
          opacity: expanded ? 1 : 0,
          transition: 'opacity 0.2s ease',
          lineHeight: 1,
        }}
      >
        {info.label}
      </Text>
    </Box>
  )

  if (forceExpand) {
    return badge
  }

  return (
    <Tooltip label={info.label} withArrow position="top" fz="xs" disabled={expanded}>
      {badge}
    </Tooltip>
  )
}

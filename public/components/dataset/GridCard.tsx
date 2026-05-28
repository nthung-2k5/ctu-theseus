import { Box, Checkbox, Image, Paper, Text, Tooltip } from '@mantine/core'
import type { DatasetImage } from '@public/store/types'
import { useState } from 'react'
import { SplitBadgeIcon } from './SplitBadgeIcon'

function ClassSquare({ cls }: { cls?: { name: string; color: string } }) {
  if (!cls) return null
  return (
    <Tooltip label={cls.name} withArrow position="top" fz="xs">
      <Box w={16} h={16} bdrs={3} bg={cls.color} bd="2px solid white" />
    </Tooltip>
  )
}

interface GridCardProps {
  image: DatasetImage
  selected: boolean
  onSelect: (id: string, checked: boolean) => void
  cls?: { name: string; color: string }
}

export function GridCard({ image, selected, onSelect, cls }: GridCardProps) {
  const [hovered, setHovered] = useState(false)

  return (
    <Paper
      radius="sm"
      withBorder
      style={{
        overflow: 'hidden',
        cursor: 'pointer',
        outline: selected ? '2px solid var(--mantine-primary-color-5)' : '2px solid transparent',
        outlineOffset: -2,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onSelect(image.id, !selected)}
    >
      {/* Image area */}
      <Box pos="relative" style={{ aspectRatio: '4/3', overflow: 'hidden' }}>
        <Image src={`/${image.path}`} alt={image.filename} fit="cover" w="100%" h="100%" display="block" />

        {/* Hover darkening on the cell border, not the image */}
        {hovered && (
          <Box
            pos="absolute"
            inset={0}
            className="pointer-events-none z-3"
            style={{ boxShadow: 'inset 0 0 0 3px rgba(0,0,0,0.3)' }}
          />
        )}

        {/* Top-right: checkbox on hover or selected */}
        {(hovered || selected) && (
          <Box pos="absolute" top={4} right={4} className="z-4">
            <Checkbox
              checked={selected}
              onChange={(e) => {
                e.stopPropagation()
                onSelect(image.id, e.currentTarget.checked)
              }}
              onClick={(e) => e.stopPropagation()}
              size="xs"
              styles={{
                input: { cursor: 'pointer' },
              }}
            />
          </Box>
        )}

        {/* Bottom-left: split icon */}
        <Box pos="absolute" bottom={4} left={4} className="z-2">
          <SplitBadgeIcon split={image.split} />
        </Box>

        {/* Bottom-right: class color square */}
        <Box pos="absolute" bottom={4} right={4} className="z-2">
          <ClassSquare cls={cls} />
        </Box>
      </Box>

      {/* Filename caption */}
      <Box px="xs" py={6}>
        <Text size="xs" truncate="end" c="dimmed">
          {image.filename}
        </Text>
      </Box>
    </Paper>
  )
}

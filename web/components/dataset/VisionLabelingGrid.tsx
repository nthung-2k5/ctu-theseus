import { ActionIcon, Badge, Box, Card, Checkbox, Group, Image, Switch, Text } from '@mantine/core'
import { CheckCircleIcon } from '@phosphor-icons/react'
import type { Annotation, LabelClass } from '@public/store/types'
import { useState } from 'react'
import { ClassPalette } from './ClassPalette'
import { useAssignAnnotation } from './useAssignAnnotation'
import { findClassificationAnnotation } from './utils'

interface VisionItem {
  id: string
  downloadUrl: string | null
  annotations?: Annotation[]
}

/**
 * Gallery grid for image classification: click a thumbnail to select it (or
 * check its box to add it to a multi-select), then click a class in the
 * palette to assign it. Digit keys 1-9 assign the currently-focused item
 * directly, without needing to click the palette.
 */
export function VisionLabelingGrid<T extends VisionItem>({
  projectId,
  items,
  classes,
}: {
  projectId: string
  items: T[]
  classes: LabelClass[]
}) {
  const assign = useAssignAnnotation(projectId)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [focused, setFocused] = useState<string | null>(null)
  const [unlabeledOnly, setUnlabeledOnly] = useState(false)
  const classNameById = new Map(classes.map((c) => [c.classId, c.name]))

  const visibleItems = unlabeledOnly ? items.filter((i) => !findClassificationAnnotation(i.annotations)) : items

  const assignTo = (itemIds: string[], classId: string) => {
    for (const itemId of itemIds) {
      const item = items.find((i) => i.id === itemId)
      const existing = item && findClassificationAnnotation(item.annotations)
      assign.mutate({ itemId, existingAnnotationId: existing?.id, classId })
    }
  }

  const toggleSelect = (itemId: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(itemId)) next.delete(itemId)
      else next.add(itemId)
      return next
    })
  }

  const paletteTargets = selected.size > 0 ? [...selected] : focused ? [focused] : []

  return (
    <div>
      <Group justify="space-between" mb="sm">
        <Switch
          label="Unlabeled only"
          checked={unlabeledOnly}
          onChange={(e) => setUnlabeledOnly(e.currentTarget.checked)}
          size="sm"
        />
        {paletteTargets.length > 0 && (
          <Group gap="xs">
            <Text size="xs" c="dimmed">
              {paletteTargets.length > 1 ? `${paletteTargets.length} selected —` : 'Assign:'}
            </Text>
            <ClassPalette classes={classes} onSelect={(classId) => assignTo(paletteTargets, classId)} size="xs" />
          </Group>
        )}
      </Group>

      <Box
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
          gap: 'var(--mantine-spacing-sm)',
        }}
      >
        {visibleItems.map((item) => {
          const annotation = findClassificationAnnotation(item.annotations)
          const className = annotation?.classId ? classNameById.get(annotation.classId) : undefined
          const isSelected = selected.has(item.id)
          return (
            <Card
              key={item.id}
              withBorder
              p={0}
              radius="md"
              className="card-elevated"
              style={{
                cursor: 'pointer',
                outline: isSelected
                  ? '2px solid var(--mantine-primary-color-5)'
                  : focused === item.id
                    ? '2px solid var(--mantine-color-gray-5)'
                    : undefined,
                outlineOffset: -2,
                overflow: 'hidden',
              }}
              onClick={() => setFocused(item.id)}
            >
              <Box pos="relative">
                <Image src={item.downloadUrl ?? undefined} alt="" h={110} fit="cover" fallbackSrc="" />
                <Checkbox
                  checked={isSelected}
                  onChange={() => toggleSelect(item.id)}
                  onClick={(e) => e.stopPropagation()}
                  pos="absolute"
                  top={6}
                  left={6}
                  size="xs"
                />
                {className && (
                  <ActionIcon pos="absolute" top={4} right={4} size="sm" color="teal" variant="filled" radius="xl">
                    <CheckCircleIcon size={14} weight="fill" />
                  </ActionIcon>
                )}
              </Box>
              <Box p={6}>
                <Badge
                  size="xs"
                  variant={className ? 'light' : 'outline'}
                  color={className ? 'teal' : 'gray'}
                  fullWidth
                >
                  {className ?? 'Unlabeled'}
                </Badge>
              </Box>
            </Card>
          )
        })}
      </Box>
    </div>
  )
}

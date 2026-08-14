import { Badge, Checkbox, Group, Stack, Switch, Text } from '@mantine/core'
import { DataTable, type DataTableColumn } from '@public/components/ui'
import type { Annotation, LabelClass } from '@public/store/types'
import type { ReactNode } from 'react'
import { useState } from 'react'
import { ClassPalette } from './ClassPalette'
import { useAssignAnnotation } from './useAssignAnnotation'
import { findClassificationAnnotation } from './utils'

interface ListItem {
  id: string
  annotations?: Annotation[]
}

/**
 * Row-oriented labeling surface shared by text and audio — same
 * select/assign/unlabeled-filter interaction as VisionLabelingGrid, just a
 * table instead of a thumbnail grid. `renderContent` is the one thing that
 * differs per modality (a text excerpt vs. an <audio> player).
 */
export function LabelingList<T extends ListItem>({
  projectId,
  items,
  classes,
  renderContent,
  contentHeader,
  isLoading,
}: {
  projectId: string
  items: T[]
  classes: LabelClass[]
  renderContent: (item: T) => ReactNode
  contentHeader: string
  isLoading?: boolean
}) {
  const assign = useAssignAnnotation(projectId)
  const [selected, setSelected] = useState<Set<string>>(new Set())
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

  const columns: DataTableColumn<T>[] = [
    {
      key: 'select',
      header: '',
      render: (item) => <Checkbox checked={selected.has(item.id)} onChange={() => toggleSelect(item.id)} size="xs" />,
    },
    { key: 'content', header: contentHeader, render: renderContent },
    {
      key: 'label',
      header: 'Label',
      render: (item) => {
        const annotation = findClassificationAnnotation(item.annotations)
        const className = annotation?.classId ? classNameById.get(annotation.classId) : undefined
        return (
          <Badge size="xs" variant={className ? 'light' : 'outline'} color={className ? 'teal' : 'gray'}>
            {className ?? 'Unlabeled'}
          </Badge>
        )
      },
    },
    {
      key: 'assign',
      header: 'Assign',
      render: (item) => (
        <ClassPalette classes={classes} onSelect={(classId) => assignTo([item.id], classId)} active={false} size="xs" />
      ),
    },
  ]

  return (
    <Stack gap="sm">
      <Group justify="space-between">
        <Switch
          label="Unlabeled only"
          checked={unlabeledOnly}
          onChange={(e) => setUnlabeledOnly(e.currentTarget.checked)}
          size="sm"
        />
        {selected.size > 0 && (
          <Group gap="xs">
            <Text size="xs" c="dimmed">
              {selected.size} selected —
            </Text>
            <ClassPalette classes={classes} onSelect={(classId) => assignTo([...selected], classId)} size="xs" />
          </Group>
        )}
      </Group>
      <DataTable
        columns={columns}
        data={visibleItems}
        getRowKey={(item) => item.id}
        loading={isLoading}
        emptyMessage="No items to label."
      />
    </Stack>
  )
}

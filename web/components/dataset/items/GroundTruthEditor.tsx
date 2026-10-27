import { Box, Button, Group, NumberInput, Select, Text } from '@mantine/core'
import { CheckIcon } from '@phosphor-icons/react'
import { useState } from 'react'
import { CaptionEditor } from './CaptionEditor'
import { findGroundTruth, type GroundTruthSpec, groundTruthKind, targetValue } from './groundTruth'
import { TagEditor } from './TagEditor'
import type { CaptionEditing, DatasetListItem, ItemEditing } from './types'

/** A regression target: one number per item, stored as `labelStructured.value`. */
const TargetEditor = ({
  item,
  spec,
  editing,
}: {
  item: DatasetListItem
  spec: GroundTruthSpec
  editing?: ItemEditing
}) => {
  const existing = findGroundTruth(item.annotations, spec)
  const [value, setValue] = useState<number | string>(targetValue(existing) ?? '')
  const [saved, setSaved] = useState<number | null>(targetValue(existing))
  const [saving, setSaving] = useState(false)

  if (!editing) {
    return (
      <Text size="sm" c={saved === null ? 'dimmed' : undefined}>
        {saved ?? 'No target yet'}
      </Text>
    )
  }

  const next = typeof value === 'number' && Number.isFinite(value) ? value : null
  const handleSave = async () => {
    if (next === null) return
    setSaving(true)
    try {
      await editing.onSaveTarget(item.id, existing?.id ?? null, next)
      setSaved(next)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Box>
      <NumberInput label="Target" value={value} onChange={setValue} decimalScale={10} allowNegative />
      <Group justify="flex-end" mt="xs">
        <Button
          size="xs"
          leftSection={<CheckIcon size={14} />}
          onClick={handleSave}
          loading={saving}
          disabled={next === null || next === saved}
        >
          Save
        </Button>
      </Group>
    </Box>
  )
}

/**
 * The label section of the item drawer. What it renders is decided by the task registry's annotation spec:
 * a class picker, a numeric target, free text, or a word-by-word tag editor.
 */
export const GroundTruthEditor = ({
  item,
  spec,
  classNameById,
  classColorById,
  captionEditing,
  editing,
}: {
  item: DatasetListItem
  spec: GroundTruthSpec
  classNameById: Map<string, string>
  classColorById?: Map<string, string>
  captionEditing?: CaptionEditing
  editing?: ItemEditing
}) => {
  const current = findGroundTruth(item.annotations, spec)

  switch (groundTruthKind(spec)) {
    case 'class': {
      const classId = current?.classId ?? null
      if (!editing || editing.classes.length === 0) {
        return (
          <Group justify="space-between" py={4}>
            <Text size="sm" c="dimmed">
              Label
            </Text>
            <Text size="sm">{classNameById.get(classId ?? '') ?? 'Unassigned'}</Text>
          </Group>
        )
      }
      return (
        <Select
          label="Label"
          placeholder="Unassigned"
          data={editing.classes.map((c) => ({ value: c.classId, label: c.name }))}
          value={classId}
          allowDeselect={false}
          onChange={(next) => next && next !== classId && editing.onSetClass(item.id, next)}
        />
      )
    }
    case 'number':
      return <TargetEditor item={item} spec={spec} editing={editing} />
    case 'text':
      return captionEditing && editing ? (
        <CaptionEditor item={item} captionEditing={captionEditing} />
      ) : (
        <Box>
          <Text size="sm" c="dimmed">
            {captionEditing?.label ?? 'Label'}
          </Text>
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
            {current?.labelTextSequence || '—'}
          </Text>
        </Box>
      )
    case 'tags':
      // Keyed by item: the editor holds the tags being typed, which must not carry over to the next item.
      return (
        <TagEditor
          key={item.id}
          item={item}
          stored={current?.labelStructured}
          storedId={current?.id ?? null}
          editing={editing}
          classColorById={classColorById}
        />
      )
    default:
      return null
  }
}

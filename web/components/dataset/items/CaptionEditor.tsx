import { Box, Button, Group, Textarea } from '@mantine/core'
import { CheckIcon } from '@phosphor-icons/react'
import { useState } from 'react'
import { type CaptionEditing, type DatasetListItem, findTextSequenceAnnotation } from './types'

export const CaptionEditor = ({ item, captionEditing }: { item: DatasetListItem; captionEditing: CaptionEditing }) => {
  const existing = findTextSequenceAnnotation(item.annotations)
  const [text, setText] = useState(existing?.labelTextSequence ?? '')
  const [saving, setSaving] = useState(false)
  // Tracks what's actually persisted, separately from `item.annotations`
  // (a snapshot taken when the modal opened) — so "Save" correctly disables
  // itself right after a successful save instead of staying enabled until
  // the modal is closed and reopened against a refetched item.
  const [savedText, setSavedText] = useState(existing?.labelTextSequence ?? '')

  const handleSave = async () => {
    setSaving(true)
    try {
      await captionEditing.onSave(item.id, existing?.id ?? null, text)
      setSavedText(text)
    } finally {
      setSaving(false)
    }
  }

  const dirty = text !== savedText

  return (
    <Box mt="md">
      <Textarea
        label={captionEditing.label}
        placeholder={`Enter a ${captionEditing.label.toLowerCase()}…`}
        value={text}
        onChange={(e) => setText(e.currentTarget.value)}
        autosize
        minRows={2}
        maxRows={6}
      />
      <Group justify="flex-end" mt="xs">
        <Button size="xs" leftSection={<CheckIcon size={14} />} onClick={handleSave} loading={saving} disabled={!dirty}>
          Save
        </Button>
      </Group>
    </Box>
  )
}

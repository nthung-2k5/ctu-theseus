import { Button, Group, Stack, Text, Textarea } from '@mantine/core'
import { type DraftClass, parseBulk } from '@public/lib/classDraft'
import { useState } from 'react'

/**
 * Add several classes from pasted text (one per line, or separated by commas or semicolons). Shows what will be
 * added and what is left out (duplicates, over-long names) before anything changes. Used in a dialog, and inline
 * on the empty page where a dialog would be one click too many.
 */
export function BulkAddForm({
  draft,
  initialText = '',
  onAdd,
  onCancel,
}: {
  draft: DraftClass[]
  initialText?: string
  onAdd: (names: string[]) => void
  onCancel?: () => void
}) {
  const [text, setText] = useState(initialText)
  const { names, skipped } = parseBulk(text, draft)

  return (
    <Stack>
      <Textarea
        autosize
        minRows={5}
        maxRows={10}
        aria-label="Class names"
        placeholder="One per line, or comma-separated"
        value={text}
        onChange={(e) => setText(e.currentTarget.value)}
        data-autofocus
      />
      <Text size="xs" c="dimmed">
        {names.length} to add
        {skipped.length ? ` · skipped (duplicate or too long): ${skipped.join(', ')}` : ''}
      </Text>
      <Group justify="flex-end">
        {onCancel && (
          <Button variant="default" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button
          disabled={names.length === 0}
          onClick={() => {
            onAdd(names)
            setText('')
          }}
        >
          Add {names.length}
        </Button>
      </Group>
    </Stack>
  )
}

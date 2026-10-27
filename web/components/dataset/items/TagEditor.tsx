import { Alert, Box, Button, Group, Text, UnstyledButton } from '@mantine/core'
import { CheckIcon, WarningIcon } from '@phosphor-icons/react'
import { useState } from 'react'
import {
  fillUntagged,
  isComplete,
  isValidTag,
  OUTSIDE,
  sameTags,
  setRange,
  setTag,
  splitWords,
  strayInside,
  taggedCount,
  tagsForWords,
} from './tagEditing'
import type { DatasetListItem, ItemEditing } from './types'

const NEUTRAL = 'var(--mantine-color-default-border)'

/**
 * Tags a text word by word: pick a tag, then click the words that have it (shift-click for a stretch of words).
 * A tag is one of the project's classes, so the Classes tab is where the tag set lives; a class name with a space in it
 * cannot be a tag (a snapshot joins tags with spaces) and is not offered. Only a text with a tag on every word can be
 * saved, because the server wants exactly one per word and training reads nothing from a half-tagged text.
 */
export const TagEditor = ({
  item,
  stored,
  storedId,
  editing,
  classColorById,
}: {
  item: DatasetListItem
  /** The annotation's `labelStructured` as it was when the drawer opened. */
  stored: unknown
  storedId: string | null
  editing?: ItemEditing
  classColorById?: Map<string, string>
}) => {
  const words = splitWords(item.textFeatures?.rawText ?? '')
  const [saved, setSaved] = useState(() => tagsForWords(stored, words.length))
  const [tags, setTags] = useState(saved)
  const [annotationId, setAnnotationId] = useState(storedId)
  const [active, setActive] = useState<string | null>(null)
  const [anchor, setAnchor] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)

  const options = (editing?.classes ?? []).filter((c) => isValidTag(c.name))
  const colorOf = (tag: string) => options.find((c) => c.name === tag)?.uiColorHex ?? classColorById?.get(tag) ?? null
  const hasOutside = options.some((c) => c.name === OUTSIDE)

  if (words.length === 0) {
    return (
      <Text size="sm" c="dimmed">
        This item has no text to tag.
      </Text>
    )
  }

  const pick = (index: number, extend: boolean) => {
    if (!editing || active === null) return
    if (extend && anchor !== null) {
      setTags(setRange(tags, anchor, index, active))
    } else {
      // Clicking a word that already has the active tag takes it off again.
      setTags(setTag(tags, index, tags[index] === active ? '' : active))
    }
    setAnchor(index)
  }

  const complete = isComplete(tags)
  const dirty = !sameTags(tags, saved)
  const stray = strayInside(tags)
  const handleSave = async () => {
    if (!editing || !complete) return
    setSaving(true)
    try {
      // After the first save the annotation exists; a second save must update it, not create another.
      setAnnotationId(await editing.onSaveTags(item.id, annotationId, tags))
      setSaved(tags)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Box>
      <Text size="sm" c="dimmed" mb={4}>
        Tags
      </Text>

      {editing &&
        (options.length === 0 ? (
          <Alert color="yellow" icon={<WarningIcon size={16} />} mb="xs" p="xs">
            Add the tags as classes in the Classes tab first (a tag has no spaces, for example O, B-PER, I-PER).
          </Alert>
        ) : (
          <Group gap={6} mb="xs" role="group" aria-label="Tag to apply">
            {options.map((c) => {
              const color = c.uiColorHex ?? null
              const on = active === c.name
              return (
                <Button
                  key={c.classId}
                  size="compact-xs"
                  variant={on ? 'filled' : 'default'}
                  color={color ?? undefined}
                  aria-pressed={on}
                  onClick={() => setActive(on ? null : c.name)}
                >
                  {c.name}
                </Button>
              )
            })}
          </Group>
        ))}

      <Group gap={6} align="flex-start" role="group" aria-label="Words">
        {words.map((word, i) => {
          const tag = tags[i]
          const color = tag ? colorOf(tag) : null
          return (
            <UnstyledButton
              // Words repeat, and the order never changes while the item is open, so the index is the identity.
              // biome-ignore lint/suspicious/noArrayIndexKey: see above
              key={i}
              disabled={!editing}
              onClick={(e) => pick(i, e.shiftKey)}
              aria-label={`${word}: ${tag || 'no tag'}`}
              style={{
                border: `1px solid ${color ?? NEUTRAL}`,
                borderRadius: 'var(--mantine-radius-sm)',
                padding: '2px 6px',
                textAlign: 'center',
                cursor: editing && active !== null ? 'pointer' : 'default',
                background: color ? `color-mix(in srgb, ${color} 14%, transparent)` : undefined,
              }}
            >
              <Text size="sm" component="span" display="block">
                {word}
              </Text>
              <Text size="xs" component="span" display="block" c={tag ? undefined : 'dimmed'} fw={tag ? 600 : 400}>
                {tag || '—'}
              </Text>
            </UnstyledButton>
          )
        })}
      </Group>

      {stray.length > 0 && (
        <Text size="xs" c="yellow.7" mt="xs">
          {stray.length === 1 ? `“${words[stray[0]]}” is` : `${stray.length} words are`} tagged I- without a B- or I- of
          the same type just before it.
        </Text>
      )}

      {editing && (
        <Group justify="space-between" mt="xs">
          <Text size="xs" c="dimmed">
            {taggedCount(tags)} of {words.length} words tagged
          </Text>
          <Group gap="xs">
            {hasOutside && (
              <Button
                size="xs"
                variant="default"
                onClick={() => setTags(fillUntagged(tags, OUTSIDE))}
                disabled={complete}
              >
                Rest are {OUTSIDE}
              </Button>
            )}
            <Button
              size="xs"
              variant="default"
              onClick={() => setTags(new Array<string>(words.length).fill(''))}
              disabled={taggedCount(tags) === 0}
            >
              Clear
            </Button>
            <Button
              size="xs"
              leftSection={<CheckIcon size={14} />}
              onClick={handleSave}
              loading={saving}
              disabled={!complete || !dirty}
            >
              Save
            </Button>
          </Group>
        </Group>
      )}
    </Box>
  )
}

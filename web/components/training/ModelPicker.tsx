import { Badge, Group, ScrollArea, Stack, Text, TextInput, UnstyledButton } from '@mantine/core'
import { MagnifyingGlassIcon } from '@phosphor-icons/react'
import type { ModelChoiceOut, TrainingBackendOut } from '@public/lib/api/generated/models'
import { useState } from 'react'

/** What "the model" is, in words for someone who has never trained one. */
export const MODEL_HELP =
  'The base model you fine-tune on your data. A pretrained one has already learned general patterns from a huge collection of examples, so it usually needs far less of yours.'

/** Where a model comes from, in the order the list shows them. */
const MODEL_SOURCES = [
  { source: 'builtin', label: 'Base models' },
  { source: 'global', label: 'Shared imported models' },
  { source: 'private', label: 'My imported models' },
] as const

/** A model list long enough to be worth searching. */
const SEARCH_FROM = 6

function ModelRow({ model, selected, onSelect }: { model: ModelChoiceOut; selected: boolean; onSelect: () => void }) {
  return (
    <UnstyledButton
      onClick={onSelect}
      role="radio"
      aria-checked={selected}
      p="xs"
      style={{
        borderRadius: 'var(--mantine-radius-sm)',
        border: `1px solid ${selected ? 'var(--mantine-color-primary-6)' : 'var(--mantine-color-default-border)'}`,
        background: selected ? 'var(--mantine-color-default-hover)' : 'transparent',
      }}
    >
      <Group gap={6} wrap="nowrap">
        <Text size="sm" fw={500} truncate style={{ minWidth: 0 }}>
          {model.label}
        </Text>
        <Badge size="xs" variant="outline" color="gray" style={{ flexShrink: 0 }}>
          {model.pretrained ? 'pretrained' : 'from scratch'}
        </Badge>
        {model.source === 'private' && (
          <Badge size="xs" variant="light" color="grape" style={{ flexShrink: 0 }}>
            imported
          </Badge>
        )}
        {model.source === 'global' && (
          <Badge size="xs" variant="light" color="blue" style={{ flexShrink: 0 }}>
            shared
          </Badge>
        )}
      </Group>
      {model.description && (
        <Text size="xs" c="dimmed" lineClamp={2}>
          {model.description}
        </Text>
      )}
    </UnstyledButton>
  )
}

/**
 * The base-model chooser: a scrolling, searchable list of the selected backend's models, each with its
 * description beside its pretrained / imported / shared badges. Base models come first, then an
 * administrator's shared imports, then the user's own; with only one kind the list stays flat.
 */
export function ModelPicker({
  backend,
  modelId,
  onChange,
  height = 300,
}: {
  backend: TrainingBackendOut
  modelId: string | null
  onChange: (id: string) => void
  height?: number
}) {
  const [search, setSearch] = useState('')
  const needle = search.trim().toLowerCase()
  const groups = MODEL_SOURCES.map((s) => ({
    label: s.label,
    models: backend.models
      .filter((m) => (m.source ?? 'builtin') === s.source)
      .filter((m) => !needle || `${m.label} ${m.description}`.toLowerCase().includes(needle)),
  })).filter((g) => g.models.length > 0)

  return (
    <Stack gap="xs">
      {backend.models.length > SEARCH_FROM && (
        <TextInput
          size="xs"
          placeholder="Search base models"
          aria-label="Search base models"
          leftSection={<MagnifyingGlassIcon size={14} />}
          value={search}
          onChange={(e) => setSearch(e.currentTarget.value)}
        />
      )}
      <ScrollArea h={height} type="auto" offsetScrollbars>
        <Stack gap={6} role="radiogroup" aria-label="Base model">
          {groups.length === 0 && (
            <Text size="xs" c="dimmed">
              No base model matches “{search}”.
            </Text>
          )}
          {groups.map((g) => (
            <Stack key={g.label} gap={6}>
              {groups.length > 1 && (
                <Text size="xs" c="dimmed" tt="uppercase" fw={500} mt={4}>
                  {g.label}
                </Text>
              )}
              {g.models.map((m) => (
                <ModelRow key={m.id} model={m} selected={m.id === modelId} onSelect={() => onChange(m.id)} />
              ))}
            </Stack>
          ))}
        </Stack>
      </ScrollArea>
    </Stack>
  )
}

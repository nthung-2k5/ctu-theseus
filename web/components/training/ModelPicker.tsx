import { Badge, Group, Modal, ScrollArea, Stack, Text, TextInput, UnstyledButton } from '@mantine/core'
import { CaretRightIcon, MagnifyingGlassIcon } from '@phosphor-icons/react'
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

/** A model together with the trainer that trains it: choosing a model is what chooses the trainer. */
export interface ModelEntry {
  backend: TrainingBackendOut
  model: ModelChoiceOut
}

export interface ModelSelection {
  backendId: string
  modelId: string
}

/** The id of the stand-in entry for a backend that offers no choice of model (its one model is fixed by the task). */
export const FIXED_MODEL_ID = ''

/**
 * Every backend's models as one list. A model id is only unique within its backend, so identify one by both.
 * A backend with no selectable models still gets one entry, or choosing it would be impossible.
 */
export function allModels(backends: TrainingBackendOut[]): ModelEntry[] {
  return backends.flatMap((backend) =>
    backend.models.length > 0
      ? backend.models.map((model) => ({ backend, model }))
      : [
          {
            backend,
            model: {
              id: FIXED_MODEL_ID,
              label: 'Default model',
              description: 'The model for this task is chosen for you.',
              pretrained: false,
              source: 'builtin' as const,
            },
          },
        ],
  )
}

function ModelBadges({ model }: { model: ModelChoiceOut }) {
  return (
    <>
      {model.id !== FIXED_MODEL_ID && (
        <Badge size="xs" variant="outline" color="gray" style={{ flexShrink: 0 }}>
          {model.pretrained ? 'pretrained' : 'from scratch'}
        </Badge>
      )}
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
    </>
  )
}

function ModelRow({ entry, selected, onSelect }: { entry: ModelEntry; selected: boolean; onSelect: () => void }) {
  const { model } = entry
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
        <ModelBadges model={model} />
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
 * The base-model field of the new-run form: shows the chosen model and opens a modal to change it. The modal
 * lists the models of every trainer together (searchable, base models first, then an administrator's shared
 * imports, then the user's own). Which trainer a model belongs to is deliberately not shown: it follows from
 * the model chosen.
 */
export function ModelPicker({
  backends,
  selection,
  onChange,
  builtInOnly,
}: {
  backends: TrainingBackendOut[]
  selection: ModelSelection | null
  onChange: (selection: ModelSelection) => void
  /** Leave out imported (shared or private) models, for a place that can only use the built-in ones. */
  builtInOnly?: boolean
}) {
  const [opened, setOpened] = useState(false)
  const [search, setSearch] = useState('')

  const entries = allModels(backends).filter((e) => !builtInOnly || (e.model.source ?? 'builtin') === 'builtin')
  const current = entries.find((e) => e.backend.id === selection?.backendId && e.model.id === selection?.modelId)

  const needle = search.trim().toLowerCase()
  const groups = MODEL_SOURCES.map((s) => ({
    label: s.label,
    entries: entries
      .filter((e) => (e.model.source ?? 'builtin') === s.source)
      .filter((e) => !needle || `${e.model.label} ${e.model.description}`.toLowerCase().includes(needle)),
  })).filter((g) => g.entries.length > 0)

  const close = () => {
    setOpened(false)
    setSearch('')
  }
  const choose = (entry: ModelEntry) => {
    onChange({ backendId: entry.backend.id, modelId: entry.model.id })
    close()
  }

  return (
    <>
      <UnstyledButton
        onClick={() => setOpened(true)}
        aria-haspopup="dialog"
        aria-label={current ? `Base model: ${current.model.label}. Change` : 'Choose a base model'}
        p="xs"
        w="100%"
        style={{
          borderRadius: 'var(--mantine-radius-sm)',
          border: '1px solid var(--mantine-color-default-border)',
        }}
      >
        <Group gap="xs" wrap="nowrap" justify="space-between">
          <div style={{ minWidth: 0 }}>
            {current ? (
              <>
                <Group gap={6} wrap="nowrap">
                  <Text size="sm" fw={500} truncate style={{ minWidth: 0 }}>
                    {current.model.label}
                  </Text>
                  <ModelBadges model={current.model} />
                </Group>
                {current.model.description && (
                  <Text size="xs" c="dimmed" lineClamp={2}>
                    {current.model.description}
                  </Text>
                )}
              </>
            ) : (
              <Text size="sm" c="dimmed">
                Choose a base model
              </Text>
            )}
          </div>
          <CaretRightIcon size={14} style={{ flexShrink: 0 }} />
        </Group>
      </UnstyledButton>

      <Modal opened={opened} onClose={close} title="Choose a base model" size="lg" centered>
        <Stack gap="xs">
          {entries.length > SEARCH_FROM && (
            <TextInput
              size="xs"
              placeholder="Search base models"
              aria-label="Search base models"
              leftSection={<MagnifyingGlassIcon size={14} />}
              data-autofocus
              value={search}
              onChange={(e) => setSearch(e.currentTarget.value)}
            />
          )}
          <ScrollArea h={460} type="auto" offsetScrollbars>
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
                  {g.entries.map((e) => (
                    <ModelRow
                      key={`${e.backend.id}:${e.model.id}`}
                      entry={e}
                      selected={e.backend.id === selection?.backendId && e.model.id === selection?.modelId}
                      onSelect={() => choose(e)}
                    />
                  ))}
                </Stack>
              ))}
            </Stack>
          </ScrollArea>
        </Stack>
      </Modal>
    </>
  )
}

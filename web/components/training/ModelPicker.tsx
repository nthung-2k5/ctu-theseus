import { Badge, Group, Select, Text } from '@mantine/core'
import type { TrainingBackendOut } from '@public/lib/api/generated/models'

/** What "the model" is, in words for someone who has never trained one. */
export const MODEL_HELP =
  'The model that learns from your data. A pretrained one has already learned general patterns from a huge collection of examples, so it usually needs far less of yours.'

/** Where a model comes from, in the order the picker lists them. */
const MODEL_SOURCES = [
  { source: 'builtin', label: 'Built-in' },
  { source: 'global', label: 'Shared by your organization' },
  { source: 'private', label: 'My models' },
] as const

/**
 * The model chooser: a searchable list of the selected backend's models, with who it is shared by and its
 * description beside it. Used by the Simple block canvas and by Advanced, so the choice is never only one click
 * away in the other view.
 */
export function ModelPicker({
  backend,
  modelId,
  onChange,
}: {
  backend: TrainingBackendOut
  modelId: string | null
  onChange: (id: string) => void
}) {
  const modelInfo = backend.models.find((m) => m.id === modelId)
  // Built-in models first, then an administrator's shared ones, then the user's own. With no custom
  // models the picker stays the flat list it always was.
  const modelGroups = MODEL_SOURCES.map((s) => ({
    group: s.label,
    items: backend.models
      .filter((m) => (m.source ?? 'builtin') === s.source)
      .map((m) => ({ value: m.id, label: m.label })),
  })).filter((g) => g.items.length > 0)
  const modelOptions =
    modelGroups.length > 1 ? modelGroups : backend.models.map((m) => ({ value: m.id, label: m.label }))

  return (
    <Group gap="md" align="flex-start">
      <Select
        size="xs"
        w={260}
        aria-label="Model"
        allowDeselect={false}
        searchable
        data={modelOptions}
        value={modelId}
        onChange={(v) => v && onChange(v)}
      />
      <Group gap={6} align="flex-start" wrap="nowrap" pt={6}>
        {modelInfo && modelInfo.source !== undefined && modelInfo.source !== 'builtin' && (
          <Badge size="xs" variant="light" color={modelInfo.source === 'private' ? 'grape' : 'blue'}>
            {modelInfo.source === 'private' ? 'my model' : 'shared'}
          </Badge>
        )}
        <Text size="xs" c="dimmed" maw={420}>
          {modelInfo?.description}
        </Text>
      </Group>
    </Group>
  )
}

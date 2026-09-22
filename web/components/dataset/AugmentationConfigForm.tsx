/**
 * Augmentation picker for the "Create Snapshot" dialog.
 *
 * Nothing here knows about a specific augmentation: the list of ops and every
 * op's tunable parameters come from GET /projects/:id/augmentations, where
 * each op is a Python class (ai_service/theseus/augmentation/ops/). A new op
 * class shows up in this form after a backend restart with no frontend change.
 */

import { Checkbox, Group, NumberInput, Select, Slider, Stack, Switch, Text } from '@mantine/core'
import type { AugmentationConfig, AugmentationInfo, ParamSpec } from '@public/lib/api/generated/models'

export interface AugmentationDraft {
  copiesPerItem: number
  /** Selected ops by id; an op that is absent is not selected. */
  ops: Record<string, { probability: number; params: Record<string, unknown> }>
}

export const emptyAugmentationDraft = (): AugmentationDraft => ({ copiesPerItem: 1, ops: {} })

const defaultParams = (op: AugmentationInfo): Record<string, unknown> =>
  Object.fromEntries(op.params.map((p) => [p.name, p.default]))

/** The request body for the selected ops, or undefined when nothing is selected. */
export function toAugmentationConfig(draft: AugmentationDraft): AugmentationConfig | undefined {
  const ops = Object.entries(draft.ops).map(([id, o]) => ({ id, probability: o.probability, params: o.params }))
  return ops.length > 0 ? { copiesPerItem: draft.copiesPerItem, ops } : undefined
}

function ParamInput({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec
  value: unknown
  onChange: (value: unknown) => void
}) {
  if (spec.type === 'bool') {
    return (
      <Switch
        size="xs"
        label={spec.label}
        description={spec.description}
        checked={Boolean(value)}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
    )
  }
  if (spec.type === 'choice') {
    return (
      <Select
        size="xs"
        label={spec.label}
        description={spec.description}
        data={spec.choices ?? []}
        value={String(value ?? '')}
        onChange={(v) => v !== null && onChange(v)}
        allowDeselect={false}
      />
    )
  }
  return (
    <NumberInput
      size="xs"
      label={spec.label}
      description={spec.description}
      value={typeof value === 'number' ? value : ''}
      min={spec.min ?? undefined}
      max={spec.max ?? undefined}
      step={spec.step ?? undefined}
      allowDecimal={spec.type === 'float'}
      decimalScale={spec.type === 'float' ? 3 : 0}
      clampBehavior="strict"
      // Ignore the transient empty string while typing: only real numbers reach the request.
      onChange={(v) => typeof v === 'number' && onChange(v)}
    />
  )
}

export function AugmentationConfigForm({
  options,
  draft,
  onChange,
  trainCount,
}: {
  options: AugmentationInfo[]
  draft: AugmentationDraft
  onChange: (draft: AugmentationDraft) => void
  /** Train-split items in the draft, for the size estimate. */
  trainCount: number
}) {
  const toggle = (op: AugmentationInfo, selected: boolean) => {
    const ops = { ...draft.ops }
    if (selected) ops[op.id] = { probability: 0.5, params: defaultParams(op) }
    else delete ops[op.id]
    onChange({ ...draft, ops })
  }

  const patch = (id: string, change: Partial<AugmentationDraft['ops'][string]>) =>
    onChange({ ...draft, ops: { ...draft.ops, [id]: { ...draft.ops[id], ...change } } })

  const selectedCount = Object.keys(draft.ops).length
  const added = trainCount * draft.copiesPerItem

  return (
    <Stack gap="sm">
      <NumberInput
        label="Copies per training item"
        description="Each training item gets this many augmented copies. Validation and test items are never augmented."
        value={draft.copiesPerItem}
        min={1}
        max={10}
        clampBehavior="strict"
        allowDecimal={false}
        onChange={(v) => typeof v === 'number' && onChange({ ...draft, copiesPerItem: v })}
        w={260}
      />

      <Stack gap="xs">
        {options.map((op) => {
          const selected = draft.ops[op.id]
          return (
            <Stack key={op.id} gap={6}>
              <Checkbox
                label={op.label}
                description={op.description}
                checked={!!selected}
                onChange={(e) => toggle(op, e.currentTarget.checked)}
              />
              {selected && (
                <Stack gap="xs" pl={28}>
                  <div>
                    <Text size="xs" fw={500}>
                      Applied to {Math.round(selected.probability * 100)}% of copies
                    </Text>
                    <Slider
                      size="sm"
                      min={0}
                      max={1}
                      step={0.05}
                      value={selected.probability}
                      label={(v) => `${Math.round(v * 100)}%`}
                      onChange={(v) => patch(op.id, { probability: v })}
                    />
                  </div>
                  {op.params.length > 0 && (
                    <Group gap="sm" align="flex-end" wrap="wrap">
                      {op.params.map((p) => (
                        <ParamInput
                          key={p.name}
                          spec={p}
                          value={selected.params[p.name]}
                          onChange={(v) => patch(op.id, { params: { ...selected.params, [p.name]: v } })}
                        />
                      ))}
                    </Group>
                  )}
                </Stack>
              )}
            </Stack>
          )
        })}
      </Stack>

      <Text size="xs" c="dimmed">
        {selectedCount === 0
          ? 'Select at least one augmentation.'
          : `Adds up to ${added.toLocaleString()} augmented item${added === 1 ? '' : 's'} (${trainCount.toLocaleString()} training item${trainCount === 1 ? '' : 's'} × ${draft.copiesPerItem}). Copies an augmentation cannot change are skipped.`}
      </Text>
    </Stack>
  )
}

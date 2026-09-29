/**
 * Preprocessing picker for the snapshot builder.
 *
 * Nothing here knows about a specific preprocessing op: the list of ops and every op's tunable
 * parameters come from GET /projects/:id/preprocessing, where each op is a Python class
 * (ai_service/theseus/preprocessing/ops/). A new op class shows up in this form after a backend
 * restart with no frontend change.
 *
 * Unlike augmentation, an op here is deterministic and REPLACES its original for whichever splits
 * it is scoped to (see the split chips below), rather than adding a probability-gated copy.
 */

import { Checkbox, Chip, Group, Paper, Stack, Text } from '@mantine/core'
import { SPLIT_TYPES } from '@public/components/dataset/VersionBrowsing'
import { ParamField } from '@public/components/ui'
import type {
  PreprocessingConfig,
  PreprocessingInfo,
  PreprocessingOpConfigSplitsItem,
} from '@public/lib/api/generated/models'

type SplitName = PreprocessingOpConfigSplitsItem

export interface PreprocessingDraft {
  /** Selected ops by id; an op that is absent is not selected. */
  ops: Record<string, { splits: SplitName[]; params: Record<string, unknown> }>
}

export const emptyPreprocessingDraft = (): PreprocessingDraft => ({ ops: {} })

const defaultParams = (op: PreprocessingInfo): Record<string, unknown> =>
  Object.fromEntries(op.params.map((p) => [p.name, p.default]))

/** The request body for the selected ops, or undefined when nothing is selected. */
export function toPreprocessingConfig(draft: PreprocessingDraft): PreprocessingConfig | undefined {
  const ops = Object.entries(draft.ops).map(([id, o]) => ({ id, splits: o.splits, params: o.params }))
  return ops.length > 0 ? { ops } : undefined
}

/** Draft-independent: the splits at least one selected op runs on, for the builder's size estimate. */
export function preprocessingSplits(draft: PreprocessingDraft): Set<SplitName> {
  return new Set(Object.values(draft.ops).flatMap((o) => o.splits))
}

const cap = (s: string) => `${s[0].toUpperCase()}${s.slice(1)}`

export function PreprocessingConfigForm({
  options,
  draft,
  onChange,
}: {
  options: PreprocessingInfo[]
  draft: PreprocessingDraft
  onChange: (draft: PreprocessingDraft) => void
}) {
  const toggle = (op: PreprocessingInfo, selected: boolean) => {
    const ops = { ...draft.ops }
    if (selected) ops[op.id] = { splits: [...SPLIT_TYPES], params: defaultParams(op) }
    else delete ops[op.id]
    onChange({ ...draft, ops })
  }

  const patch = (id: string, change: Partial<PreprocessingDraft['ops'][string]>) =>
    onChange({ ...draft, ops: { ...draft.ops, [id]: { ...draft.ops[id], ...change } } })

  const toggleSplit = (id: string, split: SplitName, on: boolean) => {
    const current = draft.ops[id].splits
    const next = on ? [...current, split] : current.filter((s) => s !== split)
    if (next.length > 0) patch(id, { splits: next })
  }

  const selectedCount = Object.keys(draft.ops).length

  return (
    <Stack gap="sm">
      <Stack gap="xs">
        {options.map((op) => {
          const selected = draft.ops[op.id]
          return (
            <Paper key={op.id} p="sm" style={selected ? { borderColor: 'var(--mantine-color-teal-5)' } : undefined}>
              <Stack gap={6}>
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
                        Replaces the original in
                      </Text>
                      <Group gap={6}>
                        {SPLIT_TYPES.map((s) => (
                          <Chip
                            key={s}
                            size="xs"
                            value={s}
                            checked={selected.splits.includes(s)}
                            onChange={(on) => toggleSplit(op.id, s, on)}
                          >
                            {cap(s)}
                          </Chip>
                        ))}
                      </Group>
                    </div>
                    {op.params.length > 0 && (
                      <Group gap="sm" align="flex-end" wrap="wrap">
                        {op.params.map((p) => (
                          <ParamField
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
            </Paper>
          )
        })}
      </Stack>

      <Text size="xs" c="dimmed">
        {selectedCount === 0
          ? 'Select at least one preprocessing op.'
          : 'Items in the splits above are replaced by their preprocessed copy; nothing is added. Copies an op cannot change are left as the original.'}
      </Text>
    </Stack>
  )
}

/**
 * Renders a plugin's tunable parameters (a `ParamSpec[]`) as form inputs, without knowing
 * anything about the plugin itself. Backs both the augmentation op picker (Create Snapshot) and
 * the trainer backend hyperparameter form (Create Run / Create Sweep) — the same wire shape
 * (`GET /projects/:id/augmentations`, `GET /projects/:id/training-backends`) drives both.
 */

import { Group, NumberInput, Select, Switch } from '@mantine/core'
import type { ParamSpec } from '@public/lib/api/generated/models'

export function ParamField({
  spec,
  value,
  onChange,
  size = 'xs',
}: {
  spec: ParamSpec
  value: unknown
  onChange: (value: unknown) => void
  size?: 'xs' | 'sm'
}) {
  if (spec.type === 'bool') {
    return (
      <Switch
        size={size}
        label={spec.label}
        description={spec.description}
        checked={Boolean(value)}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
    )
  }
  if (spec.type === 'choice') {
    // A spec whose own declared default is null/unset (e.g. Ludwig's optional "optimizer" knob)
    // means "let the backend decide" is a valid choice, so the field is clearable back to that;
    // one with a concrete default (an augmentation op's params are always initialized to theirs)
    // must always hold some value.
    const optional = spec.default == null
    return (
      <Select
        size={size}
        label={spec.label}
        description={spec.description}
        data={spec.choices ?? []}
        value={value != null ? String(value) : null}
        placeholder={optional ? 'Default' : undefined}
        onChange={(v) => onChange(v)}
        allowDeselect={optional}
        clearable={optional}
      />
    )
  }
  return (
    <NumberInput
      size={size}
      label={spec.label}
      description={spec.description}
      value={typeof value === 'number' ? value : ''}
      min={spec.min ?? undefined}
      max={spec.max ?? undefined}
      step={spec.step ?? undefined}
      allowDecimal={spec.type === 'float'}
      decimalScale={spec.type === 'float' ? 6 : 0}
      clampBehavior="strict"
      // Ignore the transient empty string while typing: only real numbers reach the request.
      onChange={(v) => typeof v === 'number' && onChange(v)}
    />
  )
}

/** One `ParamField` per spec, laid out in a wrapping row. `values`/`onChange` are keyed by `spec.name`. */
export function ParamFields({
  specs,
  values,
  onChange,
}: {
  specs: ParamSpec[]
  values: Record<string, unknown>
  onChange: (name: string, value: unknown) => void
}) {
  if (specs.length === 0) return null
  return (
    <Group gap="sm" align="flex-end" wrap="wrap">
      {specs.map((spec) => (
        <ParamField key={spec.name} spec={spec} value={values[spec.name]} onChange={(v) => onChange(spec.name, v)} />
      ))}
    </Group>
  )
}

/**
 * CreateSweepPanel – expand a search space into N trials and dispatch them all at once.
 *
 * Mirrors CreateRunPanel's backend-driven knob list (GET /projects/:id/training-backends is still
 * the single source of truth for what's tunable), but each knob here takes a LIST of candidate
 * values instead of one — see ai_service/theseus/services/sweep.py for how that list is expanded
 * into trials. A `bool`-typed knob has no sweepable candidate list (there's nothing to gain from
 * running a run per pydantic-declared boolean when the model choice/batch size are usually the
 * more interesting axes) and is left out of this form; every `int`/`float`/`choice` knob is.
 */

import {
  Button,
  Card,
  Checkbox,
  Group,
  MultiSelect,
  NumberInput,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { FlaskIcon } from '@phosphor-icons/react'
import type { ParamSpec } from '@public/lib/api/generated/models'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import type { ProjectDetail, SweepSearchSpace, SweepStrategyValue } from '@public/store/types'
import { useEffect, useState } from 'react'

export interface SweepStartConfig {
  name: string
  datasetVersionId: string
  backend: string
  searchSpace: SweepSearchSpace
  strategy: SweepStrategyValue
  maxTrials: number
}

function parseNumberList(raw: string): number[] {
  return raw
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n))
}

/** One knob's checkbox + candidate-value input, sharing the "included in the search space" state. */
function SweepKnobField({
  spec,
  included,
  onIncludedChange,
  value,
  onValueChange,
}: {
  spec: ParamSpec
  included: boolean
  onIncludedChange: (included: boolean) => void
  value: string | string[]
  onValueChange: (value: string | string[]) => void
}) {
  return (
    <Group gap="sm" align="flex-start">
      <Checkbox mt={30} checked={included} onChange={(e) => onIncludedChange(e.currentTarget.checked)} />
      {spec.type === 'choice' ? (
        <MultiSelect
          flex={1}
          label={`${spec.label} candidates`}
          data={spec.choices ?? []}
          disabled={!included}
          value={Array.isArray(value) ? value : []}
          onChange={onValueChange}
        />
      ) : (
        <TextInput
          flex={1}
          label={`${spec.label} candidates`}
          description={spec.type === 'int' ? 'Comma-separated integers' : 'Comma-separated numbers'}
          disabled={!included}
          value={typeof value === 'string' ? value : ''}
          onChange={(e) => onValueChange(e.currentTarget.value)}
        />
      )}
    </Group>
  )
}

export function CreateSweepPanel({
  project,
  onStartSweep,
}: {
  project: ProjectDetail
  onStartSweep: (config: SweepStartConfig) => void
}) {
  const dataset = project.dataset
  const { data } = useListProjectTrainingBackends(project.id)
  const backends = data?.backends ?? []

  const [backendId, setBackendId] = useState<string | null>(null)
  useEffect(() => {
    if (!backendId && backends.length > 0) setBackendId(backends[0].id)
  }, [backendId, backends])
  const backend = backends.find((b) => b.id === backendId)
  const sweepableKnobs = (backend?.params ?? []).filter((p) => p.type !== 'bool')

  const [included, setIncluded] = useState<Record<string, boolean>>({})
  const [candidates, setCandidates] = useState<Record<string, string | string[]>>({})
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only when the backend itself changes, not on every knob edit
  useEffect(() => {
    setIncluded({})
    setCandidates(
      Object.fromEntries(
        sweepableKnobs.map((p) => [p.name, p.type === 'choice' ? [] : p.default != null ? String(p.default) : '']),
      ),
    )
  }, [backend])

  const versionOptions =
    dataset?.versions
      ?.filter((v) => v.status === 'ready')
      .map((v) => ({ value: v.id, label: `${v.versionTag} (${v.itemCount ?? 0} items)` })) ?? []

  const form = useForm({
    initialValues: { name: '', datasetVersionId: '', strategy: 'grid' as SweepStrategyValue, maxTrials: 6 },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Sweep name is required'),
      datasetVersionId: (v) => (v ? null : 'Please select a ready dataset snapshot'),
    },
  })

  const includedCount = Object.values(included).filter(Boolean).length

  const handleSubmit = (values: typeof form.values) => {
    if (!backend) return
    const searchSpace: SweepSearchSpace = {}
    for (const spec of sweepableKnobs) {
      if (!included[spec.name]) continue
      const raw = candidates[spec.name]
      if (spec.type === 'choice') {
        if (Array.isArray(raw) && raw.length > 0) searchSpace[spec.name] = raw
      } else if (typeof raw === 'string') {
        const numbers = parseNumberList(raw)
        if (numbers.length > 0) searchSpace[spec.name] = numbers
      }
    }

    onStartSweep({
      name: values.name,
      datasetVersionId: values.datasetVersionId,
      backend: backend.id,
      searchSpace,
      strategy: values.strategy,
      maxTrials: values.maxTrials,
    })
  }

  return (
    <Card withBorder p="lg" radius="md">
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Stack gap="lg">
          <Group gap="sm">
            <FlaskIcon size={24} />
            <Title order={4}>New Sweep</Title>
          </Group>

          <Text size="sm" c="dimmed">
            Check the knobs to search over and list their candidate values. Every combination (grid) or a random sample
            of combinations (random) is dispatched as its own training run.
          </Text>

          <TextInput label="Sweep Name" placeholder="e.g. Encoder + LR search" {...form.getInputProps('name')} />

          <Select
            label="Dataset Snapshot"
            placeholder={
              versionOptions.length === 0
                ? 'No ready snapshots — create one on the Dataset page'
                : 'Select a snapshot to train on'
            }
            data={versionOptions}
            disabled={versionOptions.length === 0}
            {...form.getInputProps('datasetVersionId')}
            searchable
          />

          {backends.length > 1 && (
            <Select
              label="Trainer Backend"
              data={backends.map((b) => ({ value: b.id, label: b.label }))}
              value={backendId}
              onChange={setBackendId}
              allowDeselect={false}
            />
          )}

          <Group grow align="flex-end">
            <Select
              label="Strategy"
              data={[
                { value: 'grid', label: 'Grid — every combination' },
                { value: 'random', label: 'Random — sample combinations' },
              ]}
              allowDeselect={false}
              {...form.getInputProps('strategy')}
            />
            <NumberInput
              label="Max Trials"
              description="Caps how many runs this sweep can dispatch"
              min={1}
              max={50}
              {...form.getInputProps('maxTrials')}
            />
          </Group>

          <Stack gap="sm">
            {sweepableKnobs.map((spec) => (
              <SweepKnobField
                key={spec.name}
                spec={spec}
                included={!!included[spec.name]}
                onIncludedChange={(v) => setIncluded((prev) => ({ ...prev, [spec.name]: v }))}
                value={candidates[spec.name] ?? (spec.type === 'choice' ? [] : '')}
                onValueChange={(v) => setCandidates((prev) => ({ ...prev, [spec.name]: v }))}
              />
            ))}
          </Stack>

          {includedCount === 0 && (
            <Text size="xs" c="red">
              Check at least one hyperparameter to search over.
            </Text>
          )}

          <Group justify="flex-end">
            <Button
              type="submit"
              leftSection={<FlaskIcon size={16} />}
              disabled={versionOptions.length === 0 || includedCount === 0 || !backend}
            >
              Start Sweep
            </Button>
          </Group>
        </Stack>
      </form>
    </Card>
  )
}

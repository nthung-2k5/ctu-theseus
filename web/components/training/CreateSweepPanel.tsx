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

import { Checkbox, Group, MultiSelect, NumberInput, Paper, Select, SimpleGrid, Text, TextInput } from '@mantine/core'
import { FlaskIcon } from '@phosphor-icons/react'
import { groupParams, SectionLabel } from '@public/components/ui'
import type { ParamSpec } from '@public/lib/api/generated/models'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail, SweepSearchSpace, SweepStrategyValue } from '@public/store/types'
import { useEffect, useMemo, useState } from 'react'
import { ConfigHeader, LaunchBar } from './ConfigHeader'

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
      <Checkbox size="xs" mt={26} checked={included} onChange={(e) => onIncludedChange(e.currentTarget.checked)} />
      {spec.type === 'choice' ? (
        <MultiSelect
          size="xs"
          flex={1}
          label={`${spec.label} candidates`}
          data={spec.choices ?? []}
          disabled={!included}
          value={Array.isArray(value) ? value : []}
          onChange={onValueChange}
        />
      ) : (
        <TextInput
          size="xs"
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
  loading,
}: {
  project: ProjectDetail
  onStartSweep: (config: SweepStartConfig) => void
  loading?: boolean
}) {
  const { data } = useListProjectTrainingBackends(project.id)
  const backends = data?.backends ?? []

  const [name, setName] = useState('')
  const [strategy, setStrategy] = useState<SweepStrategyValue>('grid')
  const [maxTrials, setMaxTrials] = useState<number | string>(6)

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

  // Only ready snapshots have a parquet to train on.
  const readyVersions = useMemo(
    () => (project.dataset?.versions ?? []).filter((v) => v.status === 'ready'),
    [project.dataset?.versions],
  )
  const [versionId, setVersionId] = useState<string | null>(null)
  useEffect(() => {
    if (!versionId && readyVersions.length > 0) setVersionId(readyVersions[readyVersions.length - 1].id)
  }, [versionId, readyVersions])

  const includedCount = Object.values(included).filter(Boolean).length
  const classCount = project.dataset?.classes?.length ?? 0

  const problems: string[] = []
  if (readyVersions.length === 0) problems.push('Build a snapshot first: there is nothing to train on yet.')
  else if (!versionId) problems.push('Select a snapshot.')
  if (!name.trim()) problems.push('Give the sweep a name.')
  if (isClassificationTask(project.task) && classCount < 2) problems.push('Define at least 2 classes.')
  if (!backend) problems.push('No trainer backend is available.')
  if (includedCount === 0) problems.push('Check at least one hyperparameter to search over.')

  const launch = () => {
    if (!backend || !versionId) return
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
      name: name.trim(),
      datasetVersionId: versionId,
      backend: backend.id,
      searchSpace,
      strategy,
      maxTrials: Number(maxTrials) || 1,
    })
  }

  return (
    <div className="flex flex-col gap-3">
      <ConfigHeader
        project={project}
        nameLabel="Sweep name"
        namePlaceholder="e.g. encoder + LR search"
        name={name}
        onNameChange={setName}
        versionId={versionId}
        onVersionChange={setVersionId}
        backends={backends}
        backendId={backendId}
        onBackendChange={setBackendId}
        extras={
          <>
            <Select
              size="xs"
              label="Strategy"
              w={230}
              allowDeselect={false}
              data={[
                { value: 'grid', label: 'Grid: every combination' },
                { value: 'random', label: 'Random: sample combinations' },
              ]}
              value={strategy}
              onChange={(v) => v && setStrategy(v as SweepStrategyValue)}
            />
            <NumberInput
              size="xs"
              label="Max trials"
              w={100}
              min={1}
              max={50}
              allowDecimal={false}
              value={maxTrials}
              onChange={setMaxTrials}
            />
          </>
        }
      />

      <Paper p="sm">
        <SectionLabel mb={4}>Search space</SectionLabel>
        <Text size="xs" c="dimmed" mb="sm">
          Check the knobs to search over and list their candidate values. Every combination (grid) or a random sample of
          combinations (random) is dispatched as its own training run.
        </Text>
        <div className="flex flex-col gap-3">
          {groupParams(sweepableKnobs).map(({ group, specs }) => (
            <div key={group}>
              <Text size="xs" fw={500} mb={6}>
                {group}
              </Text>
              <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
                {specs.map((spec) => (
                  <SweepKnobField
                    key={spec.name}
                    spec={spec}
                    included={!!included[spec.name]}
                    onIncludedChange={(v) => setIncluded((prev) => ({ ...prev, [spec.name]: v }))}
                    value={candidates[spec.name] ?? (spec.type === 'choice' ? [] : '')}
                    onValueChange={(v) => setCandidates((prev) => ({ ...prev, [spec.name]: v }))}
                  />
                ))}
              </SimpleGrid>
            </div>
          ))}
        </div>
      </Paper>

      <LaunchBar
        problems={problems}
        label="Launch sweep"
        icon={<FlaskIcon size={15} />}
        loading={loading}
        onLaunch={launch}
      />
    </div>
  )
}

/**
 * CreateSweepPanel – expand a search space into N trials and dispatch them all at once.
 *
 * Mirrors CreateRunPanel's backend-driven knob list (GET /projects/:id/training-backends is still
 * the single source of truth for what's tunable), but each knob here takes a LIST of candidate
 * values instead of one — see server/omnitrain/services/sweep.py for how that list is expanded
 * into trials. A `bool`-typed knob has no sweepable candidate list (there's nothing to gain from
 * running a run per pydantic-declared boolean when the model choice/batch size are usually the
 * more interesting axes) and is left out of this form; every `int`/`float`/`choice` knob is.
 */

import { Checkbox, Group, MultiSelect, NumberInput, Paper, Select, SimpleGrid, Text, TextInput } from '@mantine/core'
import { FlaskIcon } from '@phosphor-icons/react'
import { groupParams, InfoTip, LabelWithTip, SectionLabel } from '@public/components/ui'
import type { ParamSpec } from '@public/lib/api/generated/models'
import { useListSweepStrategies } from '@public/lib/api/generated/sweeps/sweeps'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail, SweepSearchSpace, SweepStrategyValue } from '@public/store/types'
import { useEffect, useMemo, useState } from 'react'
import { ConfigHeader, LaunchBar } from './ConfigHeader'
import { allModels, MODEL_HELP, ModelPicker, type ModelSelection } from './ModelPicker'

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
          label={<LabelWithTip label={`${spec.label} candidates`} tip={spec.description} />}
          data={spec.choices ?? []}
          disabled={!included}
          value={Array.isArray(value) ? value : []}
          onChange={onValueChange}
        />
      ) : (
        <TextInput
          size="xs"
          flex={1}
          label={<LabelWithTip label={`${spec.label} candidates`} tip={spec.description} />}
          description={
            `${spec.type === 'int' ? 'Comma-separated integers' : 'Comma-separated numbers'}` +
            (spec.disabledValue != null ? `; ${spec.disabledValue} = disabled` : '')
          }
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
  const { data: strategyList } = useListSweepStrategies()
  const strategies = strategyList?.strategies ?? []
  const [strategy, setStrategy] = useState<SweepStrategyValue>('grid')
  const chosenStrategy = strategies.find((s) => s.id === strategy)
  const [maxTrials, setMaxTrials] = useState<number | string>(6)

  // The chosen model decides the trainer, and so which settings can be searched.
  const [selection, setSelection] = useState<ModelSelection | null>(null)
  useEffect(() => {
    if (selection || backends.length === 0) return
    const first = allModels(backends).find((e) => (e.model.source ?? 'builtin') === 'builtin')
    if (first) setSelection({ backendId: first.backend.id, modelId: first.model.id })
  }, [selection, backends])
  const backend = backends.find((b) => b.id === selection?.backendId)
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
  if (!backend) problems.push('No model is available to train this task.')
  if (includedCount === 0) problems.push('Check at least one hyperparameter to search over.')

  const launch = () => {
    if (!backend || !versionId) return
    // The chosen model is a search-space key like any other, with one candidate, so every trial uses it.
    const searchSpace: SweepSearchSpace = selection?.modelId ? { [backend.modelParamName]: [selection.modelId] } : {}
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
        extras={
          <>
            <Select
              size="xs"
              label={
                <LabelWithTip
                  label="Strategy"
                  tip={`How the settings you tick below are combined into trials. ${chosenStrategy?.description ?? ''}`}
                />
              }
              w={260}
              allowDeselect={false}
              data={strategies.map((s) => ({ value: s.id, label: s.label }))}
              value={strategy}
              onChange={(v) => v && setStrategy(v as SweepStrategyValue)}
            />
            <NumberInput
              size="xs"
              label={
                <LabelWithTip
                  label="Max trials"
                  tip="The most models this sweep will train. Each combination is one trial, so a long list of values can add up quickly."
                />
              }
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
        <Group gap={4} mb="xs">
          <SectionLabel>Base model</SectionLabel>
          <InfoTip text={MODEL_HELP} />
        </Group>
        <ModelPicker backends={backends} selection={selection} onChange={setSelection} builtInOnly />
      </Paper>

      <Paper p="sm">
        <SectionLabel mb={4}>Search space</SectionLabel>
        <Text size="xs" c="dimmed" mb="sm">
          Tick the settings you want to try and list the values to try for each. The strategy picks which combinations
          to train, and each one is its own run, so you can compare the results.
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

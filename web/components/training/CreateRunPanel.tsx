/**
 * CreateRunPanel – form to start a new training run.
 *
 * The models of every backend and their hyperparameters all come from
 * GET /projects/:id/training-backends. There is no separate backend choice: the models are one list
 * (opened from the "Base model" field), and the model picked decides the backend and so which settings the
 * right-hand column shows. This is what makes it a no-code trainer: the form is
 * generated from what an installed trainer backend plugin actually supports for this task, not a
 * fixed Ludwig-shaped schema baked into the frontend. Adding a backend, or changing what one
 * offers, needs no change here.
 *
 * Two columns: on the left what to train (run name, base model, snapshot), on the right how (the
 * configuration). The everyday settings (input size, head, class weights, epochs, batch size, random seed)
 * are always shown, and "Show advanced settings" reveals every other hyperparameter below
 * them. Both edit the same state, so toggling never loses anything.
 *
 * Preprocessing and augmentation are not training options: both are chosen when a snapshot is
 * created (see SnapshotBuilderPage), so the resulting items are real, browsable dataset items.
 */

import {
  Alert,
  Button,
  Collapse,
  Group,
  Paper,
  SegmentedControl,
  SimpleGrid,
  Skeleton,
  Text,
  TextInput,
} from '@mantine/core'
import { CaretDownIcon, CaretUpIcon, WarningIcon } from '@phosphor-icons/react'
import { groupParams, InfoTip, LabelWithTip, ParamField, SectionLabel } from '@public/components/ui'
import type { TrainingBackendOut } from '@public/lib/api/generated/models'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { BasicSettings, basicParamNames } from './BasicSettings'
import { LaunchBar, RUN_NAME_HELP } from './ConfigHeader'
import { FIXED_MODEL_ID, MODEL_HELP, ModelPicker, type ModelSelection } from './ModelPicker'
import { SNAPSHOT_HELP, SnapshotPicker } from './SnapshotPicker'

export interface RunPrefill {
  name: string
  datasetVersionId: string
  backend?: string
  hyperparameters: Record<string, unknown>
}

interface CreateRunPanelProps {
  project: ProjectDetail
  /** Start from an earlier run's setup ("New run from this setup"). */
  prefill?: RunPrefill
  loading?: boolean
  onStartTraining: (config: {
    name: string
    datasetVersionId: string
    backend: string
    hyperparameters: unknown
  }) => void
}

/** The settings always shown next to the basics, when the backend has them. */
const QUICK_PARAM_NAMES = ['epochs', 'batchSize', 'seed']

export function CreateRunPanel({ project, prefill, loading, onStartTraining }: CreateRunPanelProps) {
  const { data, isLoading } = useListProjectTrainingBackends(project.id)
  const backends = useMemo(() => data?.backends ?? [], [data?.backends])

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [name, setName] = useState('')
  const [versionId, setVersionId] = useState<string | null>(null)

  // The chosen model decides the trainer backend: there is no separate backend choice.
  const [selection, setSelection] = useState<ModelSelection | null>(null)
  const backend = backends.find((b) => b.id === selection?.backendId)
  const modelId = selection?.modelId || null

  const [values, setValues] = useState<Record<string, unknown>>({})
  // Start from a backend's defaults, overlaid with the prefill's values for knobs that backend actually has.
  const startWith = useCallback((b: TrainingBackendOut, from?: Record<string, unknown>) => {
    const defaults = Object.fromEntries(b.params.map((p) => [p.name, p.default]))
    const overlay = Object.fromEntries(Object.entries(from ?? {}).filter(([k]) => k in defaults))
    setValues({ ...defaults, ...overlay })
    const wanted = from?.[b.modelParamName]
    const model = b.models.find((m) => m.id === wanted) ?? b.models[0]
    setSelection({ backendId: b.id, modelId: model?.id ?? FIXED_MODEL_ID })
  }, [])
  // On first load, and again when a source run's setup arrives: its backend (or, failing that, the one whose
  // catalog holds its model), else the first backend.
  useEffect(() => {
    if (backends.length === 0) return
    const from = prefill?.hyperparameters
    const owner =
      backends.find((b) => b.id === prefill?.backend) ??
      backends.find((b) => b.models.some((m) => m.id === from?.[b.modelParamName]))
    startWith(owner ?? backends[0], from)
  }, [backends, prefill, startWith])

  // Only snapshots that finished building can actually be trained on: the draft is mutable and has
  // no parquet, and `building`/`failed` versions have none either.
  const readyVersions = useMemo(
    () => (project.dataset?.versions ?? []).filter((v) => v.status === 'ready'),
    [project.dataset?.versions],
  )
  useEffect(() => {
    if (versionId || readyVersions.length === 0) return
    const preferred = prefill && readyVersions.some((v) => v.id === prefill.datasetVersionId)
    setVersionId(preferred ? (prefill?.datasetVersionId ?? null) : readyVersions[readyVersions.length - 1].id)
  }, [versionId, readyVersions, prefill])

  useEffect(() => {
    if (prefill) setName(`${prefill.name} (copy)`)
  }, [prefill])

  const classCount = project.dataset?.classes?.length ?? 0
  const problems: string[] = []
  if (readyVersions.length === 0) problems.push('Build a snapshot first: there is nothing to train on yet.')
  else if (!versionId) problems.push('Select a snapshot.')
  if (!name.trim()) problems.push('Give the run a name.')
  if (isClassificationTask(project.task) && classCount < 2) problems.push('Define at least 2 classes.')
  if (!backend) problems.push('No model is available to train this task.')

  const params = backend?.params ?? []
  const quick = useMemo(() => {
    const named = QUICK_PARAM_NAMES.map((n) => params.find((p) => p.name === n)).filter((p) => !!p)
    return named.length > 0 ? named : params.filter((p) => p.type !== 'bool').slice(0, 2)
  }, [params])

  // Everything the basic form and the quick settings do not already show.
  const advancedParams = useMemo(() => {
    const shown = basicParamNames(params)
    return params.filter((p) => !shown.has(p.name) && !quick.some((q) => q.name === p.name))
  }, [params, quick])

  const launch = () => {
    if (!backend || !versionId) return
    onStartTraining({
      name: name.trim(),
      datasetVersionId: versionId,
      backend: backend.id,
      hyperparameters: { ...values, ...(modelId && { [backend.modelParamName]: modelId }) },
    })
  }

  if (isLoading) return <Skeleton height={320} radius="md" />

  if (backends.length === 0) {
    return (
      <Alert icon={<WarningIcon size={18} />} color="yellow" title="No model available">
        Nothing installed can currently train this project's task. Ask an administrator to check the server's setup.
      </Alert>
    )
  }

  const setValue = (n: string, v: unknown) => setValues((prev) => ({ ...prev, [n]: v }))
  // A model trained from scratch has no pretrained weights to freeze. Only for a backend that has the switch: a
  // value for a parameter it does not know would be rejected when the run is started.
  // Picking a model from another backend swaps in that backend's settings, since the two do not share them.
  const changeModel = (next: ModelSelection) => {
    const nextBackend = backends.find((b) => b.id === next.backendId)
    if (!nextBackend) return
    if (nextBackend.id !== backend?.id) startWith(nextBackend)
    setSelection(next)
    const hasFreezeSwitch = nextBackend.params.some((p) => p.name === 'freezeBackbone')
    if (hasFreezeSwitch && !nextBackend.models.find((m) => m.id === next.modelId)?.pretrained) {
      setValue('freezeBackbone', false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        {/* Left: what to train — the run's name, its base model and the snapshot it learns from. */}
        <div className="flex flex-col gap-3">
          <Paper p="sm">
            <TextInput
              size="xs"
              label={<LabelWithTip label="Run name" tip={RUN_NAME_HELP} />}
              placeholder="e.g. baseline"
              value={name}
              onChange={(e) => setName(e.currentTarget.value)}
              error={name.trim() ? undefined : 'required'}
            />
          </Paper>
          <Paper p="sm">
            <Group gap={4} mb="xs">
              <SectionLabel>Base model</SectionLabel>
              <InfoTip text={MODEL_HELP} />
            </Group>
            <ModelPicker backends={backends} selection={selection} onChange={changeModel} />
          </Paper>
          <Paper p="sm">
            <Group gap={4} mb="xs">
              <SectionLabel>Dataset snapshot</SectionLabel>
              <InfoTip text={SNAPSHOT_HELP} />
            </Group>
            <SnapshotPicker versions={readyVersions} versionId={versionId} onChange={setVersionId} />
          </Paper>
        </div>

        {/* Right: how to train it — the configuration. */}
        <div className="flex flex-col gap-3">
          {backend && <BasicSettings backend={backend} values={values} onValue={setValue} modelId={modelId} />}
          {quick.length > 0 && (
            <Paper p="sm">
              <SectionLabel mb="xs">Training</SectionLabel>
              <Group gap="lg" align="flex-end">
                {quick.map((spec) =>
                  spec.type === 'choice' && (spec.choices?.length ?? 0) <= 6 && spec.default != null ? (
                    <div key={spec.name}>
                      <Text size="xs" fw={500} mb={4}>
                        <LabelWithTip label={spec.label} tip={spec.description} />
                      </Text>
                      <SegmentedControl
                        size="xs"
                        value={values[spec.name] != null ? String(values[spec.name]) : String(spec.default)}
                        onChange={(v) => setValue(spec.name, v)}
                        data={spec.choices ?? []}
                      />
                    </div>
                  ) : (
                    // An optional number also carries its on/off switch, so it needs more room.
                    <div key={spec.name} style={{ width: spec.default == null ? 190 : 130 }}>
                      <ParamField spec={spec} value={values[spec.name]} onChange={(v) => setValue(spec.name, v)} />
                    </div>
                  ),
                )}
              </Group>
              <Text size="xs" c="dimmed" mt="xs">
                Everything else uses sensible defaults.
              </Text>
            </Paper>
          )}

          {advancedParams.length > 0 && (
            <>
              <div>
                <Button
                  size="xs"
                  variant="default"
                  aria-expanded={showAdvanced}
                  rightSection={showAdvanced ? <CaretUpIcon size={12} /> : <CaretDownIcon size={12} />}
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  {showAdvanced ? 'Hide advanced settings' : `Show advanced settings (${advancedParams.length})`}
                </Button>
              </div>
              <Collapse expanded={showAdvanced}>
                <div className="flex flex-col gap-3">
                  {groupParams(advancedParams).map(({ group, specs }) => (
                    <Paper key={group} p="sm">
                      <SectionLabel mb="xs">{group}</SectionLabel>
                      <SimpleGrid cols={{ base: 2, xl: 3 }} spacing="sm" style={{ alignItems: 'end' }}>
                        {specs.map((spec) => (
                          <ParamField
                            key={spec.name}
                            spec={spec}
                            value={values[spec.name]}
                            onChange={(v) => setValue(spec.name, v)}
                          />
                        ))}
                      </SimpleGrid>
                    </Paper>
                  ))}
                </div>
              </Collapse>
            </>
          )}
        </div>
      </div>

      <LaunchBar problems={problems} label="Launch run" loading={loading} onLaunch={launch} />
    </div>
  )
}

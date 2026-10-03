/**
 * CreateRunPanel – form to start a new training run.
 *
 * The backend choice, its selectable models and its hyperparameters all come from
 * GET /projects/:id/training-backends — this is what makes it a no-code trainer: the form is
 * generated from what an installed trainer backend plugin actually supports for this task, not a
 * fixed Ludwig-shaped schema baked into the frontend. Adding a backend, or changing what one
 * offers, needs no change here.
 *
 * One config, one form: the everyday settings (base model, input size, head, class weights, epochs,
 * batch size) are always shown, and "Show advanced settings" reveals every other hyperparameter below
 * them. Both edit the same state, so toggling never loses anything.
 *
 * Preprocessing and augmentation are not training options: both are chosen when a snapshot is
 * created (see SnapshotBuilderPage), so the resulting items are real, browsable dataset items.
 */

import { Alert, Button, Collapse, Group, Paper, SegmentedControl, SimpleGrid, Skeleton, Text } from '@mantine/core'
import { CaretDownIcon, CaretUpIcon, WarningIcon } from '@phosphor-icons/react'
import { groupParams, InfoTip, LabelWithTip, ParamField, SectionLabel } from '@public/components/ui'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'
import { useEffect, useMemo, useState } from 'react'
import { BasicSettings, basicParamNames } from './BasicSettings'
import { ConfigHeader, LaunchBar } from './ConfigHeader'
import { MODEL_HELP, ModelPicker } from './ModelPicker'
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
const QUICK_PARAM_NAMES = ['epochs', 'batchSize']

export function CreateRunPanel({ project, prefill, loading, onStartTraining }: CreateRunPanelProps) {
  const { data, isLoading } = useListProjectTrainingBackends(project.id)
  const backends = data?.backends ?? []

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [name, setName] = useState('')
  const [versionId, setVersionId] = useState<string | null>(null)

  const [backendId, setBackendId] = useState<string | null>(null)
  useEffect(() => {
    if (backendId || backends.length === 0) return
    const preferred = prefill?.backend && backends.some((b) => b.id === prefill.backend) ? prefill.backend : null
    setBackendId(preferred ?? backends[0].id)
  }, [backendId, backends, prefill?.backend])
  const backend = backends.find((b) => b.id === backendId)

  const [modelId, setModelId] = useState<string | null>(null)
  const [values, setValues] = useState<Record<string, unknown>>({})
  // Defaults first, then overlay the prefill's values for knobs this backend actually has.
  useEffect(() => {
    const defaults = Object.fromEntries((backend?.params ?? []).map((p) => [p.name, p.default]))
    const overlay = Object.fromEntries(Object.entries(prefill?.hyperparameters ?? {}).filter(([k]) => k in defaults))
    setValues({ ...defaults, ...overlay })
    const prefillModel = backend
      ? (prefill?.hyperparameters?.[backend.modelParamName] as string | undefined)
      : undefined
    setModelId(
      prefillModel && backend?.models.some((m) => m.id === prefillModel)
        ? prefillModel
        : (backend?.models[0]?.id ?? null),
    )
  }, [backend, prefill])

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
  if (!backend) problems.push('No trainer backend is available.')

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
      <Alert icon={<WarningIcon size={18} />} color="yellow" title="No trainer backend available">
        No installed trainer backend can currently train this project's task. Check the backend's optional dependencies,
        or see /api/training-backends for why each one is unavailable.
      </Alert>
    )
  }

  const setValue = (n: string, v: unknown) => setValues((prev) => ({ ...prev, [n]: v }))
  // A model trained from scratch has no pretrained weights to freeze.
  const changeModel = (id: string) => {
    setModelId(id)
    if (!backend?.models.find((m) => m.id === id)?.pretrained) setValue('freezeBackbone', false)
  }

  return (
    <div className="flex flex-col gap-3">
      <ConfigHeader
        project={project}
        nameLabel="Run name"
        namePlaceholder="e.g. baseline"
        name={name}
        onNameChange={setName}
        versionId={versionId}
        onVersionChange={setVersionId}
        backends={backends}
        backendId={backendId}
        onBackendChange={setBackendId}
        hideSnapshot
        hideClassCount
      />

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
        {backend && (
          <Paper p="sm">
            <Group gap={4} mb="xs">
              <SectionLabel>Base model</SectionLabel>
              <InfoTip text={MODEL_HELP} />
            </Group>
            <ModelPicker backend={backend} modelId={modelId} onChange={changeModel} />
          </Paper>
        )}
        <Paper p="sm">
          <Group gap={4} mb="xs">
            <SectionLabel>Dataset snapshot</SectionLabel>
            <InfoTip text={SNAPSHOT_HELP} />
          </Group>
          <SnapshotPicker versions={readyVersions} versionId={versionId} onChange={setVersionId} />
        </Paper>
      </SimpleGrid>

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
                <div key={spec.name} style={{ width: 130 }}>
                  <ParamField spec={spec} value={values[spec.name]} onChange={(v) => setValue(spec.name, v)} />
                </div>
              ),
            )}
            <Text size="xs" c="dimmed" maw={360}>
              Everything else uses sensible defaults.
            </Text>
          </Group>
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
                  <SimpleGrid cols={{ base: 2, md: 3 }} spacing="sm" style={{ alignItems: 'end' }}>
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

      <LaunchBar problems={problems} label="Launch run" loading={loading} onLaunch={launch} />
    </div>
  )
}

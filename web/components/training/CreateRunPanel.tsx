/**
 * CreateRunPanel – form to start a new training run.
 *
 * The backend choice, its selectable models and its hyperparameters all come from
 * GET /projects/:id/training-backends — this is what makes it a no-code trainer: the form is
 * generated from what an installed trainer backend plugin actually supports for this task, not a
 * fixed Ludwig-shaped schema baked into the frontend. Adding a backend, or changing what one
 * offers, needs no change here.
 *
 * One config, two views: Simple picks the model and a couple of headline settings, Advanced shows
 * every hyperparameter. Both edit the same state, so switching never loses anything.
 *
 * Augmentation is not a training option: it is chosen when a snapshot is created (see
 * SnapshotBuilderPage), so the augmented copies are real, browsable train-split items.
 */

import { Alert, Badge, Button, Group, Paper, SegmentedControl, SimpleGrid, Skeleton, Text } from '@mantine/core'
import { WarningIcon } from '@phosphor-icons/react'
import { groupParams, ParamField, SectionLabel } from '@public/components/ui'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import { isClassificationTask } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'
import { useEffect, useMemo, useState } from 'react'
import { BlockBuilder } from './BlockBuilder'
import { ConfigHeader, LaunchBar } from './ConfigHeader'

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

type Level = 'simple' | 'advanced'

/** The settings worth surfacing in Simple mode, when the backend has them. */
const QUICK_PARAM_NAMES = ['epochs', 'batchSize']

export function CreateRunPanel({ project, prefill, loading, onStartTraining }: CreateRunPanelProps) {
  const { data, isLoading } = useListProjectTrainingBackends(project.id)
  const backends = data?.backends ?? []

  const [level, setLevel] = useState<Level>('simple')
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

  const selectedVersion = readyVersions.find((v) => v.id === versionId)
  const modelInfo = backend?.models.find((m) => m.id === modelId)
  const hasHead = params.some((p) => p.name === 'headLayers')
  const headLayers = Number(values.headLayers ?? 0)
  const blockBadges = [
    values.imageSize != null ? `pre · ${values.imageSize}px` : null,
    values.maxSequenceLength != null ? `pre · ${values.maxSequenceLength} tokens` : null,
    selectedVersion?.augmentationConfig ? `augment · ${selectedVersion.augmentationConfig.ops.length} ops` : null,
    modelInfo ? `${modelInfo.label}${values.freezeBackbone === true ? ' · frozen' : ''}` : null,
    hasHead ? (headLayers === 0 ? 'head · linear' : `head · ${headLayers}×${values.headWidth ?? 256}`) : null,
    values.useClassWeights === true ? 'class weights' : null,
  ].filter((b): b is string => !!b)

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
        right={
          <SegmentedControl
            size="xs"
            value={level}
            onChange={(v) => setLevel(v as Level)}
            data={[
              { value: 'simple', label: 'Simple' },
              { value: 'advanced', label: 'Advanced' },
            ]}
          />
        }
      />

      {level === 'simple' ? (
        <>
          {backend && (
            <BlockBuilder
              backend={backend}
              values={values}
              onValue={setValue}
              modelId={modelId}
              onModelChange={setModelId}
              version={selectedVersion}
            />
          )}
          <Paper p="sm">
            <SectionLabel mb="xs">Quick settings</SectionLabel>
            <Group gap="lg" align="flex-end">
              {quick.map((spec) =>
                spec.type === 'choice' && (spec.choices?.length ?? 0) <= 6 && spec.default != null ? (
                  <div key={spec.name}>
                    <Text size="xs" fw={500} mb={4}>
                      {spec.label}
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
                Everything else uses sensible defaults. Open <b>Advanced</b> for all {params.length} hyperparameters.
              </Text>
            </Group>
          </Paper>
        </>
      ) : (
        <>
          <Paper p="sm">
            <Group justify="space-between" wrap="nowrap">
              <Group gap={6}>
                <SectionLabel>Architecture</SectionLabel>
                {blockBadges.map((b) => (
                  <Badge key={b} variant="light" color="gray" tt="none">
                    {b}
                  </Badge>
                ))}
              </Group>
              <Button size="compact-xs" variant="default" onClick={() => setLevel('simple')}>
                Edit blocks in Simple
              </Button>
            </Group>
          </Paper>
          {groupParams(params).map(({ group, specs }) => (
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
        </>
      )}

      <LaunchBar problems={problems} label="Launch run" loading={loading} onLaunch={launch} />
    </div>
  )
}

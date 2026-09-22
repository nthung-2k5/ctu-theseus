/**
 * CreateRunPanel – form to start a new training run.
 *
 * The backend choice, its selectable models and its hyperparameters all come from
 * GET /projects/:id/training-backends — this is what makes it a no-code trainer: the form is
 * generated from what an installed trainer backend plugin actually supports for this task, not a
 * fixed Ludwig-shaped schema baked into the frontend. Adding a backend, or changing what one
 * offers, needs no change here.
 *
 * Augmentation is not a training option: it is chosen when a snapshot is created (see
 * DatasetPage), so the augmented copies are real, browsable train-split items.
 */

import { Alert, Button, Card, Group, Select, Skeleton, Stack, TextInput, Title } from '@mantine/core'
import { useForm } from '@mantine/form'
import { BrainIcon, WarningIcon } from '@phosphor-icons/react'
import { ParamFields } from '@public/components/ui'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import type { ProjectDetail } from '@public/store/types'
import { useEffect, useState } from 'react'

interface CreateRunPanelProps {
  project: ProjectDetail
  onStartTraining: (config: {
    name: string
    datasetVersionId: string
    backend: string
    hyperparameters: unknown
  }) => void
}

export function CreateRunPanel({ project, onStartTraining }: CreateRunPanelProps) {
  const dataset = project.dataset
  const { data, isLoading } = useListProjectTrainingBackends(project.id)
  const backends = data?.backends ?? []

  const [backendId, setBackendId] = useState<string | null>(null)
  useEffect(() => {
    if (!backendId && backends.length > 0) setBackendId(backends[0].id)
  }, [backendId, backends])
  const backend = backends.find((b) => b.id === backendId)

  const [modelId, setModelId] = useState<string | null>(null)
  useEffect(() => {
    setModelId(backend?.models[0]?.id ?? null)
  }, [backend])

  const [values, setValues] = useState<Record<string, unknown>>({})
  useEffect(() => {
    setValues(Object.fromEntries((backend?.params ?? []).map((p) => [p.name, p.default])))
  }, [backend])

  // Only snapshots that finished building can actually be trained on — the
  // draft is mutable and has no parquet, and `building`/`failed` versions
  // have none either.
  const versionOptions =
    dataset?.versions
      ?.filter((v) => v.status === 'ready')
      .map((v) => ({
        value: v.id,
        label: `${v.versionTag} (${v.itemCount ?? 0} items${v.augmentedCount ? `, ${v.augmentedCount} augmented` : ''})`,
      })) ?? []

  const form = useForm({
    initialValues: { name: '', datasetVersionId: '' },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Run name is required'),
      datasetVersionId: (v) => (v ? null : 'Please select a ready dataset snapshot'),
    },
  })

  const handleSubmit = (formValues: typeof form.values) => {
    if (!backend) return
    onStartTraining({
      name: formValues.name,
      datasetVersionId: formValues.datasetVersionId,
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

  return (
    <Card withBorder p="lg" radius="md">
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Stack gap="lg">
          <Group gap="sm">
            <BrainIcon size={24} />
            <Title order={4}>New Training Run</Title>
          </Group>

          <TextInput label="Run Name" placeholder="e.g. ResNet-18 baseline" {...form.getInputProps('name')} />

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

          {backend && backend.models.length > 0 && (
            <Select
              label="Model"
              description={backend.models.find((m) => m.id === modelId)?.description || undefined}
              data={backend.models.map((m) => ({ value: m.id, label: m.label }))}
              value={modelId}
              onChange={setModelId}
              allowDeselect={false}
            />
          )}

          {backend && (
            <ParamFields
              specs={backend.params}
              values={values}
              onChange={(name, v) => setValues((prev) => ({ ...prev, [name]: v }))}
            />
          )}

          <Group justify="flex-end">
            <Button
              type="submit"
              leftSection={<BrainIcon size={16} />}
              disabled={versionOptions.length === 0 || !backend}
            >
              Start Training
            </Button>
          </Group>
        </Stack>
      </form>
    </Card>
  )
}

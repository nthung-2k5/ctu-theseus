/**
 * CreateRunPanel – form to start a new training run.
 *
 * The trainer knobs (epochs, batch size, learning rate, early-stop patience,
 * encoder choice) come straight from the project's task descriptor in the
 * task registry — this is what makes it a no-code trainer: the form is
 * generated from what Ludwig actually supports for this task, not a
 * free-form JSON blob the user has to know the shape of.
 */

import { Button, Card, Group, NumberInput, Select, Stack, Text, TextInput, Title } from '@mantine/core'
import { useForm } from '@mantine/form'
import { BrainIcon } from '@phosphor-icons/react'
import { useProjectStore } from '@public/store/useProjectStore'
import { getTaskDescriptor } from '@server/lib/tasks'

interface CreateRunPanelProps {
  projectId: string
  onStartTraining: (config: { name: string; datasetVersionId: string; hyperparameters: unknown }) => void
}

const BATCH_SIZE_LABEL = (v: number | 'auto') => (v === 'auto' ? 'Auto' : String(v))

export function CreateRunPanel({ projectId, onStartTraining }: CreateRunPanelProps) {
  const activeProject = useProjectStore((s) => s.activeProject)
  const dataset = activeProject?.dataset
  const descriptor = activeProject ? getTaskDescriptor(activeProject.task) : undefined
  const knobs = descriptor?.ludwig?.trainerKnobs
  const encoders = descriptor?.ludwig?.encoders ?? []

  // Only snapshots that finished building can actually be trained on — the
  // draft is mutable and has no parquet, and `building`/`failed` versions
  // have none either.
  const versionOptions =
    dataset?.versions
      ?.filter((v) => v.status === 'ready')
      .map((v) => ({ value: v.id, label: `${v.versionTag} (${v.itemCount ?? 0} items)` })) ?? []

  const form = useForm({
    initialValues: {
      name: '',
      datasetVersionId: '',
      epochs: knobs?.epochs.default ?? 20,
      batchSize: String(knobs?.batchSize.default ?? 'auto'),
      learningRate: knobs?.learningRate.default ?? 0.001,
      earlyStopPatience: knobs?.earlyStopPatience.default ?? 5,
      encoderId: encoders[0]?.id ?? '',
    },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Run name is required'),
      datasetVersionId: (v) => (v ? null : 'Please select a ready dataset snapshot'),
    },
  })

  const handleSubmit = (values: typeof form.values) => {
    onStartTraining({
      name: values.name,
      datasetVersionId: values.datasetVersionId,
      hyperparameters: {
        epochs: values.epochs,
        batchSize: values.batchSize === 'auto' ? 'auto' : Number(values.batchSize),
        learningRate: values.learningRate,
        earlyStopPatience: values.earlyStopPatience,
        ...(values.encoderId && { encoderId: values.encoderId }),
      },
    })
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
            description={dataset ? `Modality: ${dataset.modality}` : undefined}
            data={versionOptions}
            disabled={versionOptions.length === 0}
            {...form.getInputProps('datasetVersionId')}
            searchable
          />

          {encoders.length > 0 && (
            <Select
              label="Encoder"
              data={encoders.map((e) => ({ value: e.id, label: e.label }))}
              {...form.getInputProps('encoderId')}
            />
          )}

          <Group grow>
            <NumberInput
              label="Epochs"
              min={knobs?.epochs.min}
              max={knobs?.epochs.max}
              {...form.getInputProps('epochs')}
            />
            <Select
              label="Batch Size"
              data={(knobs?.batchSize.options ?? ['auto']).map((v) => ({
                value: String(v),
                label: BATCH_SIZE_LABEL(v),
              }))}
              {...form.getInputProps('batchSize')}
            />
          </Group>

          <Group grow>
            <NumberInput
              label="Learning Rate"
              min={knobs?.learningRate.min}
              max={knobs?.learningRate.max}
              decimalScale={6}
              {...form.getInputProps('learningRate')}
            />
            <NumberInput
              label="Early Stop Patience"
              description="-1 disables early stopping"
              min={knobs?.earlyStopPatience.min}
              {...form.getInputProps('earlyStopPatience')}
            />
          </Group>

          <Text size="xs" c="dimmed">
            These are the knobs {descriptor?.label ?? 'this task'} exposes for Ludwig's{' '}
            {descriptor?.ludwig?.modelType ?? 'ecd'} trainer.
          </Text>

          <Group justify="flex-end">
            <Button type="submit" leftSection={<BrainIcon size={16} />} disabled={versionOptions.length === 0}>
              Start Training
            </Button>
          </Group>
        </Stack>
      </form>
    </Card>
  )
}

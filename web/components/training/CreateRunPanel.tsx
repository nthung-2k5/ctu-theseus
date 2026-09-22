/**
 * CreateRunPanel – form to start a new training run.
 *
 * The trainer knobs (epochs, batch size, learning rate, early-stop patience,
 * encoder choice) come straight from the project's task descriptor in the
 * task registry — this is what makes it a no-code trainer: the form is
 * generated from what Ludwig actually supports for this task, not a
 * free-form JSON blob the user has to know the shape of.
 *
 * Augmentation is not a training option: it is chosen when a snapshot is
 * created (see DatasetPage), so the augmented copies are real, browsable
 * train-split items of that snapshot.
 */

import { Button, Card, Checkbox, Group, NumberInput, Select, Stack, TextInput, Title } from '@mantine/core'
import { useForm } from '@mantine/form'
import { BrainIcon } from '@phosphor-icons/react'
import { getTaskDescriptor } from '@public/lib/tasks'
import type { ProjectDetail } from '@public/store/types'

interface CreateRunPanelProps {
  project: ProjectDetail
  onStartTraining: (config: { name: string; datasetVersionId: string; hyperparameters: unknown }) => void
}

const BATCH_SIZE_LABEL = (v: number | 'auto') => (v === 'auto' ? 'Auto' : String(v))

const IMAGE_SIZE_OPTIONS = [
  { value: '128', label: '128 × 128' },
  { value: '224', label: '224 × 224' },
  { value: '256', label: '256 × 256' },
]

const OPTIMIZER_OPTIONS = [
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'sgd', label: 'SGD' },
  { value: 'rmsprop', label: 'RMSprop' },
  { value: 'adagrad', label: 'Adagrad' },
]

const CLASSIFICATION_METRIC_OPTIONS = [
  { value: 'loss', label: 'Loss' },
  { value: 'accuracy', label: 'Accuracy' },
]

const REGRESSION_METRIC_OPTIONS = [
  { value: 'loss', label: 'Loss' },
  { value: 'mean_squared_error', label: 'Mean Squared Error' },
  { value: 'mean_absolute_error', label: 'Mean Absolute Error' },
  { value: 'r2', label: 'R²' },
]

export function CreateRunPanel({ project, onStartTraining }: CreateRunPanelProps) {
  const dataset = project.dataset
  const descriptor = getTaskDescriptor(project.task)
  const knobs = descriptor.ludwig?.trainerKnobs
  const encoders = descriptor.ludwig?.encoders ?? []

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

  const isClassification = descriptor.annotation.requiresLabelClasses
  const isVision = descriptor.modality === 'vision'
  // Validation-metric choices assume a category or number output — the only
  // two output types the 'stable' (non-LLM) tasks produce; experimental
  // tasks' text/sequence outputs don't have a comparable metric menu.
  const showValidationMetric = descriptor.status === 'stable'
  const validationMetricOptions = isClassification ? CLASSIFICATION_METRIC_OPTIONS : REGRESSION_METRIC_OPTIONS

  const form = useForm({
    initialValues: {
      name: '',
      datasetVersionId: '',
      epochs: knobs?.epochs.default ?? 20,
      batchSize: String(knobs?.batchSize.default ?? 'auto'),
      learningRate: knobs?.learningRate.default ?? 0.001,
      earlyStopPatience: knobs?.earlyStopPatience.default ?? 5,
      encoderId: encoders[0]?.id ?? '',
      useClassWeights: false,
      imageSize: '',
      optimizer: '',
      validationMetric: '',
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
        ...(isClassification && values.useClassWeights && { useClassWeights: true }),
        ...(isVision && values.imageSize && { imageSize: Number(values.imageSize) }),
        ...(values.optimizer && { optimizer: values.optimizer }),
        ...(values.validationMetric && { validationMetric: values.validationMetric }),
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

          <Group grow>
            <Select
              label="Optimizer"
              placeholder="Ludwig default (Adam)"
              data={OPTIMIZER_OPTIONS}
              clearable
              {...form.getInputProps('optimizer')}
            />
            {showValidationMetric && (
              <Select
                label="Early Stop / Best-Epoch Metric"
                placeholder="Ludwig default"
                data={validationMetricOptions}
                clearable
                {...form.getInputProps('validationMetric')}
              />
            )}
          </Group>

          {isVision && (
            <Select
              label="Image Size"
              description="Resizes every training image to a square of this size"
              placeholder="Encoder default"
              data={IMAGE_SIZE_OPTIONS}
              clearable
              {...form.getInputProps('imageSize')}
            />
          )}

          {isClassification && (
            <Checkbox
              label="Weight classes by inverse frequency"
              description="Balances the loss so a minority class isn't drowned out by a majority one — recommended for imbalanced datasets"
              {...form.getInputProps('useClassWeights', { type: 'checkbox' })}
            />
          )}

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

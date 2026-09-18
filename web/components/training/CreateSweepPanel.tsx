/**
 * CreateSweepPanel – expand a search space into N trials and dispatch them
 * all at once. Mirrors CreateRunPanel's registry-driven knob list (the task
 * descriptor is still the single source of truth for what's tunable), but
 * each knob here takes a LIST of candidate values instead of one — see
 * server/lib/sweep.ts for how that list is expanded into trials.
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
import type { ProjectDetail, SweepSearchSpace, SweepStrategyValue } from '@public/store/types'
import { getTaskDescriptor } from '@server/lib/tasks'

export interface SweepStartConfig {
  name: string
  datasetVersionId: string
  searchSpace: SweepSearchSpace
  strategy: SweepStrategyValue
  maxTrials: number
}

interface SweepFormValues {
  name: string
  datasetVersionId: string
  strategy: SweepStrategyValue
  maxTrials: number
  includeEpochs: boolean
  epochsValues: string
  includeBatchSize: boolean
  batchSizeValues: string[]
  includeLearningRate: boolean
  learningRateValues: string
  includeEarlyStop: boolean
  earlyStopValues: string
  includeEncoder: boolean
  encoderValues: string[]
}

const BATCH_SIZE_LABEL = (v: number | 'auto') => (v === 'auto' ? 'Auto' : String(v))

function parseNumberList(raw: string): number[] {
  return raw
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n))
}

export function CreateSweepPanel({
  project,
  onStartSweep,
}: {
  project: ProjectDetail
  onStartSweep: (config: SweepStartConfig) => void
}) {
  const dataset = project.dataset
  const descriptor = getTaskDescriptor(project.task)
  const knobs = descriptor.ludwig?.trainerKnobs
  const encoders = descriptor.ludwig?.encoders ?? []

  const versionOptions =
    dataset?.versions
      ?.filter((v) => v.status === 'ready')
      .map((v) => ({ value: v.id, label: `${v.versionTag} (${v.itemCount ?? 0} items)` })) ?? []

  const form = useForm<SweepFormValues>({
    initialValues: {
      name: '',
      datasetVersionId: '',
      strategy: 'grid',
      maxTrials: 6,
      includeEpochs: false,
      epochsValues: String(knobs?.epochs.default ?? 20),
      includeBatchSize: false,
      batchSizeValues: [],
      includeLearningRate: true,
      learningRateValues: '0.001, 0.0001',
      includeEarlyStop: false,
      earlyStopValues: String(knobs?.earlyStopPatience.default ?? 5),
      includeEncoder: encoders.length > 0,
      encoderValues: encoders.length > 0 ? [encoders[0].id] : [],
    },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Sweep name is required'),
      datasetVersionId: (v) => (v ? null : 'Please select a ready dataset snapshot'),
    },
  })

  const noKnobsSelected =
    !form.values.includeEpochs &&
    !form.values.includeBatchSize &&
    !form.values.includeLearningRate &&
    !form.values.includeEarlyStop &&
    !form.values.includeEncoder

  const handleSubmit = (values: SweepFormValues) => {
    const searchSpace: SweepSearchSpace = {}
    if (values.includeEpochs) searchSpace.epochs = parseNumberList(values.epochsValues)
    if (values.includeBatchSize && values.batchSizeValues.length > 0) {
      searchSpace.batchSize = values.batchSizeValues.map((v) => (v === 'auto' ? 'auto' : Number(v)))
    }
    if (values.includeLearningRate) searchSpace.learningRate = parseNumberList(values.learningRateValues)
    if (values.includeEarlyStop) searchSpace.earlyStopPatience = parseNumberList(values.earlyStopValues)
    if (values.includeEncoder && values.encoderValues.length > 0) searchSpace.encoderId = values.encoderValues

    onStartSweep({
      name: values.name,
      datasetVersionId: values.datasetVersionId,
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
            <Group gap="sm" align="flex-start">
              <Checkbox mt={30} {...form.getInputProps('includeLearningRate', { type: 'checkbox' })} />
              <TextInput
                flex={1}
                label="Learning Rate candidates"
                description="Comma-separated numbers"
                disabled={!form.values.includeLearningRate}
                {...form.getInputProps('learningRateValues')}
              />
            </Group>

            {encoders.length > 0 && (
              <Group gap="sm" align="flex-start">
                <Checkbox mt={30} {...form.getInputProps('includeEncoder', { type: 'checkbox' })} />
                <MultiSelect
                  flex={1}
                  label="Encoder candidates"
                  data={encoders.map((e) => ({ value: e.id, label: e.label }))}
                  disabled={!form.values.includeEncoder}
                  {...form.getInputProps('encoderValues')}
                />
              </Group>
            )}

            <Group gap="sm" align="flex-start">
              <Checkbox mt={30} {...form.getInputProps('includeBatchSize', { type: 'checkbox' })} />
              <MultiSelect
                flex={1}
                label="Batch Size candidates"
                data={(knobs?.batchSize.options ?? ['auto']).map((v) => ({
                  value: String(v),
                  label: BATCH_SIZE_LABEL(v),
                }))}
                disabled={!form.values.includeBatchSize}
                {...form.getInputProps('batchSizeValues')}
              />
            </Group>

            <Group gap="sm" align="flex-start">
              <Checkbox mt={30} {...form.getInputProps('includeEpochs', { type: 'checkbox' })} />
              <TextInput
                flex={1}
                label="Epochs candidates"
                description="Comma-separated integers"
                disabled={!form.values.includeEpochs}
                {...form.getInputProps('epochsValues')}
              />
            </Group>

            <Group gap="sm" align="flex-start">
              <Checkbox mt={30} {...form.getInputProps('includeEarlyStop', { type: 'checkbox' })} />
              <TextInput
                flex={1}
                label="Early Stop Patience candidates"
                description="Comma-separated integers, -1 disables early stopping"
                disabled={!form.values.includeEarlyStop}
                {...form.getInputProps('earlyStopValues')}
              />
            </Group>
          </Stack>

          {noKnobsSelected && (
            <Text size="xs" c="red">
              Check at least one hyperparameter to search over.
            </Text>
          )}

          <Group justify="flex-end">
            <Button
              type="submit"
              leftSection={<FlaskIcon size={16} />}
              disabled={versionOptions.length === 0 || noKnobsSelected}
            >
              Start Sweep
            </Button>
          </Group>
        </Stack>
      </form>
    </Card>
  )
}

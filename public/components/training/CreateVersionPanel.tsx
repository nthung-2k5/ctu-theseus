import {
  Badge,
  Button,
  Card,
  ColorSwatch,
  Divider,
  Group,
  Modal,
  NumberInput,
  Progress,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { ChartPieIcon, CubeIcon, GearSixIcon, ImagesIcon, PlayIcon, QuestionIcon } from '@phosphor-icons/react'
import { useProjectClasses, useProjectDatasetStats } from '@public/queries/project'
import type { TrainingVersionConfig } from '@public/store/types'
import { useState } from 'react'
import { MODEL_FAMILIES } from './constants'

function LabelWithTooltip({ label, tooltip }: { label: string; tooltip: string }) {
  return (
    <Group gap={4} wrap="nowrap">
      <Text size="sm" fw={500}>
        {label}
      </Text>
      <Tooltip label={tooltip} multiline w={260} withArrow position="top">
        <ThemeIcon variant="subtle" color="dimmed" size="xs" radius="xl" style={{ cursor: 'help' }}>
          <QuestionIcon size={12} />
        </ThemeIcon>
      </Tooltip>
    </Group>
  )
}

interface CreateVersionPanelProps {
  projectId: string
  onStartTraining: (config: TrainingVersionConfig) => void
}

const MODEL_SELECT_DATA = MODEL_FAMILIES.map((f) => ({
  group: f.name,
  items: f.variants.map((v) => ({ value: v.id, label: v.label })),
}))

/** Find the family a variant belongs to */
function findFamily(variantId: string) {
  return MODEL_FAMILIES.find((f) => f.variants.some((v) => v.id === variantId))
}

export function CreateVersionPanel({ projectId, onStartTraining }: CreateVersionPanelProps) {
  /* ── Model state ── */
  const [variantId, setVariantId] = useState('')

  /* ── Basic hyperparams state ── */
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(16)
  const [learningRate, setLearningRate] = useState(0.001)

  /* ── Advanced hyperparams state ── */
  const [advancedMode, setAdvancedMode] = useState(false)

  // Dataset
  const [numWorkers, setNumWorkers] = useState(4)

  // Model
  const [pretrained, setPretrained] = useState(true)
  const [dropRate, setDropRate] = useState(0.0)

  // Optimization
  const [optimizer, setOptimizer] = useState<'adamw' | 'adam' | 'sgd'>('adamw')
  const [weightDecay, setWeightDecay] = useState(0.05)

  /* ── Modals ── */
  const [hyperModalOpened, { open: openHyperModal, close: closeHyperModal }] = useDisclosure()

  /* ── Dataset info from API ── */
  const { data: statsData } = useProjectDatasetStats(projectId)
  const { data: classData } = useProjectClasses(projectId)
  const classes = classData?.classes ?? []
  const totalImages = statsData?.total ?? 0
  const splitStats = {
    train: statsData?.train ?? 0,
    validation: statsData?.validation ?? 0,
    test: statsData?.test ?? 0,
    unassigned: statsData?.unassigned ?? 0,
  }

  /* ── Handlers ── */
  const handleStart = () => {
    if (!variantId) {
      notifications.show({ title: 'Missing model', message: 'Please select a model variant', color: 'orange' })
      return
    }
    if (totalImages === 0) {
      notifications.show({ title: 'No images', message: 'Upload some images to your dataset first', color: 'orange' })
      return
    }

    const config: TrainingVersionConfig = {
      dataset: {
        batch_size: batchSize,
        ...(advancedMode && { num_workers: numWorkers }),
      },
      model: {
        architecture: variantId,
        ...(advancedMode && { pretrained, drop_rate: dropRate }),
      },
      ...(advancedMode
        ? {
            optimization: {
              optimizer,
              learning_rate: learningRate,
              weight_decay: weightDecay,
            },
            schedule: {
              epochs,
            },
          }
        : {
            optimization: {
              learning_rate: learningRate,
            },
            schedule: {
              epochs,
            },
          }),
    }

    onStartTraining(config)
  }

  return (
    <Stack gap="lg">
      {/* Header */}
      <div>
        <Title order={3}>Create New Version</Title>
        <Text size="sm" c="dimmed" mt={4}>
          Configure model and training settings
        </Text>
      </div>

      {/* ── Model & Hyperparameters ── */}
      <SimpleGrid cols={2} spacing="md">
        {/* Model select */}
        <Select
          label={
            <Group gap="xs">
              <ThemeIcon variant="light" color="primary" size="xs">
                <CubeIcon size={14} />
              </ThemeIcon>
              <Text fw={600} size="sm">Model</Text>
            </Group>
          }
          placeholder="Select a model"
          data={MODEL_SELECT_DATA}
          value={variantId || null}
          onChange={(v) => setVariantId(v ?? '')}
          searchable
          nothingFoundMessage="No models found"
        />

        {/* Hyperparams card */}
        <Card
          withBorder
          p="md"
          radius="md"
          className="card-elevated"
          style={{ cursor: 'pointer' }}
          onClick={openHyperModal}
        >
          <Group gap="sm" mb="xs">
            <ThemeIcon variant="light" color="violet" size="md">
              <GearSixIcon size={18} />
            </ThemeIcon>
            <Text fw={600} size="sm">
              Hyperparameters
            </Text>
            {advancedMode && (
              <Badge size="xs" variant="light" color="violet">
                Advanced
              </Badge>
            )}
          </Group>
          <SimpleGrid cols={2} spacing={4}>
            <Text size="xs" c="dimmed">
              Epochs:{' '}
              <Text span fw={500} inherit c="var(--mantine-color-text)">
                {epochs}
              </Text>
            </Text>
            <Text size="xs" c="dimmed">
              Batch:{' '}
              <Text span fw={500} inherit c="var(--mantine-color-text)">
                {batchSize}
              </Text>
            </Text>
            <Text size="xs" c="dimmed">
              LR:{' '}
              <Text span fw={500} inherit c="var(--mantine-color-text)">
                {learningRate}
              </Text>
            </Text>
            {advancedMode && (
              <Text size="xs" c="dimmed">
                Optimizer:{' '}
                <Text span fw={500} inherit c="var(--mantine-color-text)">
                  {optimizer}
                </Text>
              </Text>
            )}
          </SimpleGrid>
        </Card>
      </SimpleGrid>

      {/* ── Dataset Overview ── */}
      <div>
        <Title order={5} mb="sm">
          Dataset Overview
        </Title>
        <Card withBorder p="md" radius="md">
          <Group justify="space-between" mb="md">
            <Group gap="sm">
              <ThemeIcon variant="light" size="lg">
                <ImagesIcon size={20} />
              </ThemeIcon>
              <div>
                <Text fw={700} size="lg">
                  {totalImages}
                </Text>
                <Text size="xs" c="dimmed">
                  Total images
                </Text>
              </div>
            </Group>
            <Group gap="sm">
              <ThemeIcon variant="light" color="secondary" size="lg">
                <ChartPieIcon size={20} />
              </ThemeIcon>
              <div>
                <Text fw={700} size="lg">
                  {classes.length}
                </Text>
                <Text size="xs" c="dimmed">
                  Classes
                </Text>
              </div>
            </Group>
          </Group>

          {totalImages > 0 && (
            <>
              <Text size="xs" c="dimmed" mb={4}>
                Split distribution
              </Text>
              <Progress.Root size="lg" radius="md">
                {splitStats.train > 0 && (
                  <Tooltip label={`Train: ${splitStats.train}`}>
                    <Progress.Section value={(splitStats.train / totalImages) * 100} color="green" />
                  </Tooltip>
                )}
                {splitStats.validation > 0 && (
                  <Tooltip label={`Validation: ${splitStats.validation}`}>
                    <Progress.Section value={(splitStats.validation / totalImages) * 100} color="blue" />
                  </Tooltip>
                )}
                {splitStats.test > 0 && (
                  <Tooltip label={`Test: ${splitStats.test}`}>
                    <Progress.Section value={(splitStats.test / totalImages) * 100} color="red" />
                  </Tooltip>
                )}
                {splitStats.unassigned > 0 && (
                  <Tooltip label={`Unassigned: ${splitStats.unassigned}`}>
                    <Progress.Section value={(splitStats.unassigned / totalImages) * 100} color="gray" />
                  </Tooltip>
                )}
              </Progress.Root>
              <Group gap="md" mt="xs">
                <Text size="xs">
                  <Badge size="xs" color="green" variant="dot" /> Train: {splitStats.train}
                </Text>
                <Text size="xs">
                  <Badge size="xs" color="blue" variant="dot" /> Val: {splitStats.validation}
                </Text>
                <Text size="xs">
                  <Badge size="xs" color="red" variant="dot" /> Test: {splitStats.test}
                </Text>
                <Text size="xs">
                  <Badge size="xs" color="gray" variant="dot" /> Unset: {splitStats.unassigned}
                </Text>
              </Group>
            </>
          )}

          {classes.length > 0 && (
            <>
              <Text size="xs" c="dimmed" mt="md" mb={4}>
                Class distribution
              </Text>
              <Group gap="xs">
                {classes.map((c) => (
                  <Badge key={c.id} variant="light" leftSection={<ColorSwatch color={c.color} size={10} />}>
                    {c.name}
                  </Badge>
                ))}
              </Group>
            </>
          )}
        </Card>
      </div>

      {/* ── Start Training ── */}
      <Button
        size="lg"
        leftSection={<PlayIcon size={20} weight="fill" />}
        onClick={handleStart}
        disabled={!variantId || totalImages === 0}
        fullWidth
      >
        Start Training
      </Button>


      {/* ── Hyperparameters Modal ── */}
      <Modal
        opened={hyperModalOpened}
        onClose={closeHyperModal}
        title="Hyperparameters"
        size={advancedMode ? 'lg' : 'md'}
        centered
      >
        <Stack gap="md">
          {/* ── Advanced toggle ── */}
          <Group justify="flex-end">
            <Switch
              label="Advanced"
              size="sm"
              checked={advancedMode}
              onChange={(e) => setAdvancedMode(e.currentTarget.checked)}
            />
          </Group>

          {/* ── Basic fields (always visible) ── */}
          <SimpleGrid cols={advancedMode ? 2 : 1} spacing="md">
            <NumberInput label={<LabelWithTooltip label="Epochs" tooltip="Number of complete passes through the entire training dataset. More epochs can improve accuracy but may overfit." />} value={epochs} onChange={(v) => setEpochs(Number(v) || 50)} min={1} max={1000} />
            <NumberInput
              label={<LabelWithTooltip label="Batch Size" tooltip="Number of samples processed before updating model weights. Larger batches train faster but use more GPU memory." />}
              value={batchSize}
              onChange={(v) => setBatchSize(Number(v) || 16)}
              min={1}
              max={256}
            />
            <NumberInput
              label={<LabelWithTooltip label="Learning Rate" tooltip="Controls how much to adjust model weights each step. Too high causes instability, too low causes slow convergence." />}
              value={learningRate}
              onChange={(v) => setLearningRate(Number(v) || 0.001)}
              min={0.00001}
              max={1}
              step={0.0001}
              decimalScale={5}
            />
          </SimpleGrid>

          {/* ── Advanced fields ── */}
          {advancedMode && (
            <>
              <Divider label="Optimization" labelPosition="center" />
              <SimpleGrid cols={2} spacing="md">
                <Select
                  label={<LabelWithTooltip label="Optimizer" tooltip="Algorithm used to update model weights. AdamW is recommended for most cases, SGD may generalize better." />}
                  data={[
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adam', label: 'Adam' },
                    { value: 'sgd', label: 'SGD' },
                  ]}
                  value={optimizer}
                  onChange={(v) => setOptimizer((v as typeof optimizer) ?? 'adamw')}
                />
                <NumberInput
                  label={<LabelWithTooltip label="Weight Decay" tooltip="L2 regularization penalty. Helps prevent overfitting by penalizing large weights. Typical values: 0.01–0.1." />}
                  value={weightDecay}
                  onChange={(v) => setWeightDecay(Number(v) || 0.05)}
                  min={0}
                  max={1}
                  step={0.01}
                  decimalScale={4}
                />
              </SimpleGrid>

              <Divider label="Model" labelPosition="center" />
              <SimpleGrid cols={2} spacing="md">
                <Switch
                  label={<LabelWithTooltip label="Pretrained Weights" tooltip="Start from weights pre-trained on ImageNet. Dramatically improves accuracy and convergence speed for most tasks." />}
                  checked={pretrained}
                  onChange={(e) => setPretrained(e.currentTarget.checked)}
                  mt="xs"
                />
                <NumberInput
                  label={<LabelWithTooltip label="Dropout Rate" tooltip="Fraction of neurons randomly disabled during training. Helps prevent overfitting. 0 = no dropout." />}
                  value={dropRate}
                  onChange={(v) => setDropRate(Number(v) || 0)}
                  min={0}
                  max={1}
                  step={0.05}
                  decimalScale={2}
                />
              </SimpleGrid>

              <Divider label="Dataset" labelPosition="center" />
              <NumberInput
                label={<LabelWithTooltip label="Data Loader Workers" tooltip="Number of CPU threads used to load and preprocess data in parallel. More workers = faster data pipeline." />}
                value={numWorkers}
                onChange={(v) => setNumWorkers(Number(v) || 4)}
                min={0}
                max={32}
              />
            </>
          )}

          <Group justify="flex-end">
            <Button variant="subtle" onClick={closeHyperModal}>
              Cancel
            </Button>
            <Button onClick={closeHyperModal}>Confirm</Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  )
}

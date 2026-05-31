import {
  Badge,
  Button,
  Card,
  Checkbox,
  ColorSwatch,
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
import { ChartPieIcon, CubeIcon, GearSixIcon, ImagesIcon, PlayIcon } from '@phosphor-icons/react'
import { useProjectClasses, useProjectDatasetStats } from '@public/queries/project'
import { useState } from 'react'
import {
  AUGMENTATION_OPTIONS,
  buildDefaultOptionState,
  MODEL_FAMILIES,
  type OptionDef,
  type OptionState,
  PREPROCESSING_OPTIONS,
  type TrainingVersionConfig,
} from './constants'

interface CreateVersionPanelProps {
  projectId: string
  onStartTraining: (config: TrainingVersionConfig) => void
}

export function CreateVersionPanel({ projectId, onStartTraining }: CreateVersionPanelProps) {
  /* ── Model state ── */
  const [modelId, setModelId] = useState(MODEL_FAMILIES[0].id)
  const [variantId, setVariantId] = useState('')

  /* ── Hyperparams state ── */
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(16)
  const [learningRate, setLearningRate] = useState(0.001)
  const [imageSize, setImageSize] = useState(224)

  /* ── Preprocessing / Augmentation state ── */
  const [preprocessing, setPreprocessing] = useState<OptionState>(() =>
    buildDefaultOptionState(PREPROCESSING_OPTIONS, ['autoOrient']),
  )
  const [augmentation, setAugmentation] = useState<OptionState>(() => buildDefaultOptionState(AUGMENTATION_OPTIONS))

  /* ── Modals ── */
  const [modelModalOpened, { open: openModelModal, close: closeModelModal }] = useDisclosure()
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

  /* ── Derived ── */
  const family = MODEL_FAMILIES.find((f) => f.id === modelId) ?? MODEL_FAMILIES[0]
  const variantLabel = family.variants.find((v) => v.id === variantId)?.label

  /* ── Handlers ── */
  const toggleOption = (type: 'preprocessing' | 'augmentation', optionId: string, enabled: boolean) => {
    const setter = type === 'preprocessing' ? setPreprocessing : setAugmentation
    setter((prev) => ({ ...prev, [optionId]: { ...prev[optionId], enabled } }))
  }

  const setOptionParam = (
    type: 'preprocessing' | 'augmentation',
    optionId: string,
    key: string,
    value: number | boolean,
  ) => {
    const setter = type === 'preprocessing' ? setPreprocessing : setAugmentation
    setter((prev) => ({
      ...prev,
      [optionId]: { ...prev[optionId], params: { ...prev[optionId].params, [key]: value } },
    }))
  }

  const handleStart = () => {
    if (!variantId) {
      notifications.show({ title: 'Missing model', message: 'Please select a model variant', color: 'orange' })
      return
    }
    if (totalImages === 0) {
      notifications.show({ title: 'No images', message: 'Upload some images to your dataset first', color: 'orange' })
      return
    }
    onStartTraining({ modelId, variantId, epochs, batchSize, learningRate, imageSize, preprocessing, augmentation })
  }

  /* ── Option card renderer ── */
  const renderOptionCard = (opt: OptionDef, type: 'preprocessing' | 'augmentation', state: OptionState) => {
    const s = state[opt.id]
    if (!s) return null
    const hasExpandedParams = s.enabled && opt.params.length > 0

    return (
      <Card key={opt.id} withBorder p="sm" radius="md">
        <Group justify="space-between" mb={hasExpandedParams ? 'xs' : 0}>
          <Text size="sm" fw={500}>
            {opt.name}
          </Text>
          <Switch size="xs" checked={s.enabled} onChange={(e) => toggleOption(type, opt.id, e.currentTarget.checked)} />
        </Group>
        {hasExpandedParams && (
          <Stack gap={6}>
            {opt.params.map((p) => (
              <Group key={p.key} gap="xs" wrap="nowrap">
                {p.type === 'boolean' ? (
                  <Checkbox
                    label={p.label}
                    size="xs"
                    checked={s.params[p.key] as boolean}
                    onChange={(e) => setOptionParam(type, opt.id, p.key, e.currentTarget.checked)}
                  />
                ) : (
                  <NumberInput
                    label={p.label}
                    size="xs"
                    value={s.params[p.key] as number}
                    onChange={(v) => setOptionParam(type, opt.id, p.key, Number(v) || (p.default as number))}
                    min={p.min}
                    max={p.max}
                    step={p.step ?? 1}
                    suffix={p.suffix}
                    style={{ flex: 1 }}
                  />
                )}
              </Group>
            ))}
          </Stack>
        )}
      </Card>
    )
  }

  return (
    <Stack gap="lg">
      {/* Header */}
      <div>
        <Title order={3}>Create New Version</Title>
        <Text size="sm" c="dimmed" mt={4}>
          Configure model, preprocessing, and augmentation settings
        </Text>
      </div>

      {/* ── Model & Hyperparameters ── */}
      <SimpleGrid cols={2} spacing="md">
        {/* Model card */}
        <Card
          withBorder
          p="md"
          radius="md"
          className="card-elevated"
          style={{ cursor: 'pointer' }}
          onClick={openModelModal}
        >
          <Group gap="sm" mb="xs">
            <ThemeIcon variant="light" color="primary" size="md">
              <CubeIcon size={18} />
            </ThemeIcon>
            <Text fw={600} size="sm">
              Model
            </Text>
          </Group>
          {variantId ? (
            <>
              <Text fw={500}>{variantLabel}</Text>
              <Text size="xs" c="dimmed">
                {family.name}
              </Text>
            </>
          ) : (
            <Text size="sm" c="dimmed">
              Click to select model
            </Text>
          )}
        </Card>

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
            <Text size="xs" c="dimmed">
              Size:{' '}
              <Text span fw={500} inherit c="var(--mantine-color-text)">
                {imageSize}px
              </Text>
            </Text>
          </SimpleGrid>
        </Card>
      </SimpleGrid>

      {/* ── Preprocessing ── */}
      <div>
        <Group gap="xs" mb="sm">
          <Title order={5}>Preprocessing</Title>
          <Badge variant="light" size="sm">
            {Object.values(preprocessing).filter((v) => v.enabled).length} active
          </Badge>
        </Group>
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="sm">
          {PREPROCESSING_OPTIONS.map((opt) => renderOptionCard(opt, 'preprocessing', preprocessing))}
        </SimpleGrid>
      </div>

      {/* ── Augmentation ── */}
      <div>
        <Group gap="xs" mb="sm">
          <Title order={5}>Augmentation</Title>
          <Badge variant="light" size="sm">
            {Object.values(augmentation).filter((v) => v.enabled).length} active
          </Badge>
        </Group>
        <SimpleGrid cols={{ base: 2, sm: 3, md: 4 }} spacing="sm">
          {AUGMENTATION_OPTIONS.map((opt) => renderOptionCard(opt, 'augmentation', augmentation))}
        </SimpleGrid>
      </div>

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

      {/* ── Model Selection Modal ── */}
      <Modal opened={modelModalOpened} onClose={closeModelModal} title="Select Model" size="lg" centered>
        <Stack gap="md">
          <Select
            label="Architecture"
            data={MODEL_FAMILIES.map((f) => ({ value: f.id, label: f.name }))}
            value={modelId}
            onChange={(v) => {
              if (v) {
                setModelId(v)
                setVariantId('')
              }
            }}
          />
          <Select
            label="Variant"
            placeholder="Choose variant"
            data={family.variants.map((v) => ({ value: v.id, label: v.label }))}
            value={variantId}
            onChange={(v) => {
              if (v) setVariantId(v)
            }}
          />
          <Group justify="flex-end">
            <Button variant="subtle" onClick={closeModelModal}>
              Cancel
            </Button>
            <Button onClick={closeModelModal} disabled={!variantId}>
              Confirm
            </Button>
          </Group>
        </Stack>
      </Modal>

      {/* ── Hyperparameters Modal ── */}
      <Modal opened={hyperModalOpened} onClose={closeHyperModal} title="Hyperparameters" centered>
        <Stack gap="md">
          <NumberInput label="Epochs" value={epochs} onChange={(v) => setEpochs(Number(v) || 50)} min={1} max={1000} />
          <NumberInput
            label="Batch Size"
            value={batchSize}
            onChange={(v) => setBatchSize(Number(v) || 16)}
            min={1}
            max={256}
          />
          <NumberInput
            label="Learning Rate"
            value={learningRate}
            onChange={(v) => setLearningRate(Number(v) || 0.001)}
            min={0.00001}
            max={1}
            step={0.0001}
            decimalScale={5}
          />
          <NumberInput
            label="Image Size"
            value={imageSize}
            onChange={(v) => setImageSize(Number(v) || 224)}
            min={32}
            max={1024}
            step={32}
            suffix="px"
          />
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

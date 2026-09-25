import { ActionIcon, Badge, Group, NumberInput, Paper, Select, Stack, Switch, Text, Tooltip } from '@mantine/core'
import { TrashIcon } from '@phosphor-icons/react'
import type { ParamSpec, TrainingBackendOut } from '@public/lib/api/generated/models'
import type { DatasetVersion } from '@public/store/types'
import { useState } from 'react'

type BlockKind = 'preprocess' | 'augment' | 'backbone' | 'head' | 'loss'

const KIND_COLOR: Record<BlockKind, string> = {
  preprocess: 'gray',
  augment: 'grape',
  backbone: 'primary',
  head: 'cyan',
  loss: 'yellow',
}

const KIND_LABEL: Record<BlockKind, string> = {
  preprocess: 'Pre',
  augment: 'Augment',
  backbone: 'Backbone',
  head: 'Head',
  loss: 'Loss',
}

/** Data flows through the pipeline in this order, so the canvas always renders in it. */
const ORDER: BlockKind[] = ['preprocess', 'augment', 'backbone', 'head', 'loss']

const DND = 'application/x-theseus-block'

/** "image_horizontal_flip" -> "horizontal flip": augmentation op ids are `<modality>_<name>`. */
const opLabel = (id: string) => id.split('_').slice(1).join(' ')

interface PaletteItem {
  id: string
  kind: BlockKind
  label: string
  /** Already on the canvas (or, for backbone and head, the variant currently selected). */
  active: boolean
  /** Backbone and head are always on the canvas: dropping another variant swaps it. */
  swap?: boolean
  apply: () => void
}

/** The pre-processing setting a backend offers for this task: image size for vision, token limit for text. */
function preprocessSetting(params: ParamSpec[]) {
  const image = params.find((p) => p.name === 'imageSize')
  if (image)
    return {
      spec: image,
      label: 'Square size (px)',
      paletteLabel: 'Image size',
      start: '224',
      hint: 'Every image is resized to a square of this size.',
    }
  const length = params.find((p) => p.name === 'maxSequenceLength')
  if (length)
    return {
      spec: length,
      label: 'Max tokens',
      paletteLabel: 'Max length',
      start: '128',
      hint: 'Longer texts are truncated to this many tokens.',
    }
  return null
}

/**
 * A block view of one run's configuration, in pipeline order: pre-processing, augmentation, the
 * backbone (the model), the head and the loss. Each block is a real setting of the selected trainer
 * backend, not a separate model of the network, so what the canvas shows is exactly what will be trained:
 *
 *   preprocess  image size (vision) or max sequence length (text)
 *   augment     the augmentation the chosen snapshot was built with (read-only: set at snapshot time)
 *   backbone    the selected model, plus a freeze switch for pretrained ones
 *   head        the fully-connected layers after the backbone: layers, width and dropout
 *   loss        class weighting (classification tasks)
 *
 * Which blocks exist depends on what the backend offers for the task, so a task without a head
 * (a fine-tuned language model, say) simply has no head block.
 */
export function BlockBuilder({
  backend,
  values,
  onValue,
  modelId,
  onModelChange,
  version,
}: {
  backend: TrainingBackendOut
  values: Record<string, unknown>
  onValue: (name: string, value: unknown) => void
  modelId: string | null
  onModelChange: (id: string) => void
  version?: DatasetVersion
}) {
  const [overIndex, setOverIndex] = useState<number | null>(null)

  const pre = preprocessSetting(backend.params)
  const layersSpec = backend.params.find((p) => p.name === 'headLayers')
  const widthSpec = backend.params.find((p) => p.name === 'headWidth')
  const dropoutSpec = backend.params.find((p) => p.name === 'headDropout')
  const freezeSpec = backend.params.find((p) => p.name === 'freezeBackbone')
  const weightsSpec = backend.params.find((p) => p.name === 'useClassWeights')
  const augmentation = version?.augmentationConfig ?? null

  const modelInfo = backend.models.find((m) => m.id === modelId)
  const preOn = !!pre && values[pre.spec.name] != null
  const headLayers = Number(values.headLayers ?? 0)
  const lossOn = !!weightsSpec && values.useClassWeights === true

  const changeModel = (id: string) => {
    onModelChange(id)
    // A model trained from scratch has no pretrained weights to freeze.
    if (!backend.models.find((m) => m.id === id)?.pretrained) onValue('freezeBackbone', false)
  }

  const palette: PaletteItem[] = [
    ...backend.models.map((m) => ({
      id: `backbone:${m.id}`,
      kind: 'backbone' as const,
      label: m.label,
      active: m.id === modelId,
      swap: true,
      apply: () => changeModel(m.id),
    })),
    ...(layersSpec
      ? [
          {
            id: 'head:linear',
            kind: 'head' as const,
            label: 'linear',
            active: headLayers === 0,
            swap: true,
            apply: () => onValue('headLayers', 0),
          },
          {
            id: 'head:mlp',
            kind: 'head' as const,
            label: 'MLP',
            active: headLayers > 0,
            swap: true,
            apply: () => headLayers === 0 && onValue('headLayers', 1),
          },
        ]
      : []),
    ...(pre
      ? [
          {
            id: 'preprocess:setting',
            kind: 'preprocess' as const,
            label: pre.paletteLabel,
            active: preOn,
            apply: () =>
              onValue(
                pre.spec.name,
                pre.spec.choices?.includes(pre.start) ? pre.start : (pre.spec.choices?.[0] ?? null),
              ),
          },
        ]
      : []),
    ...(weightsSpec
      ? [
          {
            id: 'loss:weights',
            kind: 'loss' as const,
            label: 'Class weights',
            active: lossOn,
            apply: () => onValue('useClassWeights', true),
          },
        ]
      : []),
  ]

  const blocks: BlockKind[] = ORDER.filter(
    (k) =>
      (k === 'preprocess' && preOn) ||
      (k === 'augment' && !!augmentation) ||
      k === 'backbone' ||
      (k === 'head' && !!layersSpec) ||
      (k === 'loss' && lossOn),
  )

  const canApply = (p: PaletteItem) => p.swap || !p.active

  const drop = (e: React.DragEvent) => {
    e.preventDefault()
    setOverIndex(null)
    const item = palette.find((p) => p.id === e.dataTransfer.getData(DND))
    if (item && canApply(item)) item.apply()
  }

  const gap = (i: number) => (
    // biome-ignore lint/a11y/noStaticElementInteractions: a pointer-only drop target; the palette buttons are the keyboard path
    <div
      key={`gap-${i}`}
      role="presentation"
      onDragOver={(e) => {
        e.preventDefault()
        setOverIndex(i)
      }}
      onDragLeave={() => setOverIndex((o) => (o === i ? null : o))}
      onDrop={drop}
      style={{
        height: overIndex === i ? 28 : 10,
        transition: 'height .1s',
        borderRadius: 4,
        background: overIndex === i ? 'var(--mantine-color-cyan-9)' : 'transparent',
        border: overIndex === i ? '1px dashed var(--mantine-color-cyan-5)' : 'none',
      }}
    />
  )

  const body = (kind: BlockKind) => {
    switch (kind) {
      case 'preprocess':
        return (
          <Group gap="md" align="flex-end">
            <Select
              size="xs"
              w={140}
              label={pre?.label}
              allowDeselect={false}
              data={pre?.spec.choices ?? []}
              value={pre && values[pre.spec.name] != null ? String(values[pre.spec.name]) : null}
              onChange={(v) => pre && onValue(pre.spec.name, v)}
            />
            <Text size="xs" c="dimmed">
              {pre?.hint}
            </Text>
          </Group>
        )
      case 'augment':
        return (
          <div>
            <Group gap={6}>
              {augmentation?.ops.map((op) => (
                <Badge key={op.id} color="grape" tt="none">
                  {opLabel(op.id)} · {Math.round((op.probability ?? 1) * 100)}%
                </Badge>
              ))}
            </Group>
            <Text size="xs" c="dimmed" mt={4}>
              {augmentation?.copiesPerItem ?? 1}× copies of each training item, baked into snapshot{' '}
              {version?.versionTag}. Change it by building a new snapshot.
            </Text>
          </div>
        )
      case 'backbone':
        return (
          <Stack gap="xs">
            <Group gap="md" align="flex-start">
              <Select
                size="xs"
                w={260}
                aria-label="Model"
                allowDeselect={false}
                searchable
                data={backend.models.map((m) => ({ value: m.id, label: m.label }))}
                value={modelId}
                onChange={(v) => v && changeModel(v)}
              />
              <Text size="xs" c="dimmed" maw={420} pt={6}>
                {modelInfo?.description}
              </Text>
            </Group>
            {freezeSpec && (
              <Tooltip
                label="A model trained from scratch has no pretrained weights to freeze"
                disabled={!!modelInfo?.pretrained}
              >
                <div style={{ width: 'fit-content' }}>
                  <Switch
                    size="xs"
                    label="Freeze backbone"
                    description="Train only the head; the pretrained weights stay fixed."
                    disabled={!modelInfo?.pretrained}
                    checked={values.freezeBackbone === true}
                    onChange={(e) => onValue('freezeBackbone', e.currentTarget.checked)}
                  />
                </div>
              </Tooltip>
            )}
          </Stack>
        )
      case 'head':
        return (
          <Group gap="md" align="flex-end" wrap="wrap">
            <NumberInput
              size="xs"
              w={110}
              label="Hidden layers"
              min={layersSpec?.min ?? 0}
              max={layersSpec?.max ?? 4}
              allowDecimal={false}
              clampBehavior="strict"
              value={headLayers}
              onChange={(v) => typeof v === 'number' && onValue('headLayers', v)}
            />
            {headLayers > 0 && widthSpec && (
              <Select
                size="xs"
                w={110}
                label="Width"
                allowDeselect={false}
                data={widthSpec.choices ?? []}
                value={String(values.headWidth ?? widthSpec.default)}
                onChange={(v) => v && onValue('headWidth', v)}
              />
            )}
            {headLayers > 0 && dropoutSpec && (
              <NumberInput
                size="xs"
                w={110}
                label="Dropout"
                min={dropoutSpec.min ?? 0}
                max={dropoutSpec.max ?? 0.9}
                step={dropoutSpec.step ?? 0.05}
                decimalScale={2}
                clampBehavior="strict"
                value={Number(values.headDropout ?? 0)}
                onChange={(v) => typeof v === 'number' && onValue('headDropout', v)}
              />
            )}
            <Text size="xs" c="dimmed" pb={6}>
              {headLayers === 0
                ? 'No hidden layers: the backbone feeds the output directly.'
                : `${headLayers} × ${values.headWidth ?? widthSpec?.default ?? 256} hidden units.`}
            </Text>
          </Group>
        )
      case 'loss':
        return (
          <Text size="xs" c="dimmed">
            Each class's loss is weighted inversely to how often it appears in the snapshot.
          </Text>
        )
    }
  }

  const title = (kind: BlockKind) => {
    switch (kind) {
      case 'preprocess':
        return pre?.paletteLabel.toLowerCase()
      case 'augment':
        return 'from snapshot'
      case 'backbone':
        return modelInfo?.label ?? 'model'
      case 'head':
        return headLayers === 0 ? 'linear' : 'MLP'
      case 'loss':
        return 'class weights'
    }
  }

  const remove = (kind: BlockKind) => {
    if (kind === 'preprocess' && pre) onValue(pre.spec.name, null)
    if (kind === 'loss') onValue('useClassWeights', false)
  }

  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: 'minmax(180px, 220px) 1fr' }}>
      <Paper p="xs">
        <Text size="xs" c="dimmed" tt="uppercase" mb={6}>
          Block palette
        </Text>
        <Stack gap={4}>
          {palette.map((p) => (
            <button
              type="button"
              key={p.id}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(DND, p.id)
                e.dataTransfer.effectAllowed = 'copy'
              }}
              onClick={() => canApply(p) && p.apply()}
              title={
                p.active && !p.swap
                  ? 'Already on the canvas'
                  : p.swap
                    ? 'Drag to the canvas, or click to use'
                    : 'Drag to the canvas, or click to add'
              }
              style={{
                cursor: canApply(p) ? 'grab' : 'default',
                opacity: p.active && !p.swap ? 0.45 : 1,
                textAlign: 'left',
                color: 'inherit',
                padding: '5px 8px',
                borderRadius: 4,
                fontSize: 12,
                background: p.active ? 'var(--mantine-color-dark-5)' : 'var(--mantine-color-dark-6)',
                border: `1px solid ${p.active && p.swap ? `var(--mantine-color-${KIND_COLOR[p.kind]}-6)` : 'var(--mantine-color-dark-4)'}`,
              }}
            >
              {KIND_LABEL[p.kind]} · {p.label}
            </button>
          ))}
          {!augmentation && (
            <Tooltip label="Augmentation is chosen when a snapshot is built" multiline w={200}>
              <div
                style={{
                  padding: '5px 8px',
                  borderRadius: 4,
                  fontSize: 12,
                  opacity: 0.45,
                  border: '1px dashed var(--mantine-color-dark-4)',
                }}
              >
                Augment · from snapshot
              </div>
            </Tooltip>
          )}
        </Stack>
      </Paper>

      <Stack gap={0}>
        {gap(0)}
        {blocks.map((kind, i) => (
          <div key={kind}>
            <Paper p="xs" style={{ borderLeft: `3px solid var(--mantine-color-${KIND_COLOR[kind]}-6)` }}>
              <Group justify="space-between" wrap="nowrap" align="flex-start">
                <div style={{ flex: 1 }}>
                  <Group gap={6} mb={4}>
                    <Badge color={KIND_COLOR[kind]} size="sm">
                      {kind}
                    </Badge>
                    <Text size="sm" fw={500}>
                      {title(kind)}
                    </Text>
                    {kind === 'backbone' && modelInfo && (
                      <Badge variant="outline" color="gray" size="sm">
                        {values.freezeBackbone === true
                          ? 'pretrained · frozen'
                          : modelInfo.pretrained
                            ? 'pretrained'
                            : 'from scratch'}
                      </Badge>
                    )}
                  </Group>
                  {body(kind)}
                </div>
                {(kind === 'preprocess' || kind === 'loss') && (
                  <ActionIcon
                    variant="subtle"
                    color="red"
                    size="sm"
                    aria-label="Remove block"
                    onClick={() => remove(kind)}
                  >
                    <TrashIcon size={14} />
                  </ActionIcon>
                )}
              </Group>
            </Paper>
            {gap(i + 1)}
          </div>
        ))}
      </Stack>
    </div>
  )
}

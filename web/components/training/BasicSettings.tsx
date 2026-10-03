import { Group, NumberInput, Paper, Select, Stack, Switch, Text } from '@mantine/core'
import { InfoTip, LabelWithTip, SectionLabel } from '@public/components/ui'
import type { ParamSpec, TrainingBackendOut } from '@public/lib/api/generated/models'

/** The input-size setting a backend offers for this task: image size for vision, token limit for text. */
function inputSetting(params: ParamSpec[]) {
  const image = params.find((p) => p.name === 'imageSize')
  if (image) return { spec: image, label: 'Square size (px)', hint: 'Every image is resized to a square of this size.' }
  const length = params.find((p) => p.name === 'maxSequenceLength')
  if (length) return { spec: length, label: 'Max tokens', hint: 'Longer texts are truncated to this many tokens.' }
  return null
}

/** Parameter names `BasicSettings` renders itself, so the advanced list does not repeat them. */
export function basicParamNames(params: ParamSpec[]): Set<string> {
  const names = ['headLayers', 'headWidth', 'headDropout', 'freezeBackbone', 'useClassWeights']
  const input = inputSetting(params)
  return new Set(input ? [...names, input.spec.name] : names)
}

/**
 * The everyday settings of one run as plain form controls, in the order data flows through the model:
 * the freeze switch for a pretrained base model (the model itself is chosen in its own list), the input size (vision: image size, text:
 * max sequence length), the head after the base model (layers, width, dropout) and class weighting.
 * Each is a real setting of the selected trainer backend, so what is shown is exactly what will be trained.
 * Which sections appear depends on what the backend offers for the task: one without a head setting,
 * say, has no head section.
 */
export function BasicSettings({
  backend,
  values,
  onValue,
  modelId,
}: {
  backend: TrainingBackendOut
  values: Record<string, unknown>
  onValue: (name: string, value: unknown) => void
  modelId: string | null
}) {
  const input = inputSetting(backend.params)
  const layersSpec = backend.params.find((p) => p.name === 'headLayers')
  const widthSpec = backend.params.find((p) => p.name === 'headWidth')
  const dropoutSpec = backend.params.find((p) => p.name === 'headDropout')
  const freezeSpec = backend.params.find((p) => p.name === 'freezeBackbone')
  const weightsSpec = backend.params.find((p) => p.name === 'useClassWeights')

  const modelInfo = backend.models.find((m) => m.id === modelId)
  const headLayers = Number(values.headLayers ?? 0)

  return (
    <Stack gap="sm">
      {freezeSpec && (
        <Paper p="sm">
          <Switch
            size="xs"
            label={<LabelWithTip label={freezeSpec.label} tip={freezeSpec.description} />}
            description={
              modelInfo?.pretrained ? undefined : 'Only for pretrained base models: this one is trained from scratch.'
            }
            disabled={!modelInfo?.pretrained}
            checked={values.freezeBackbone === true}
            onChange={(e) => onValue('freezeBackbone', e.currentTarget.checked)}
          />
        </Paper>
      )}

      {input && (
        <Paper p="sm">
          <Group gap="md" align="flex-end">
            <Select
              size="xs"
              w={160}
              label={<LabelWithTip label={input.label} tip={input.spec.description} />}
              placeholder="Model default"
              clearable
              data={input.spec.choices ?? []}
              value={values[input.spec.name] != null ? String(values[input.spec.name]) : null}
              onChange={(v) => onValue(input.spec.name, v)}
            />
            <Text size="xs" c="dimmed" pb={6}>
              {input.hint}
            </Text>
          </Group>
        </Paper>
      )}

      {layersSpec && (
        <Paper p="sm">
          <Stack gap="xs">
            <Group gap={4}>
              <SectionLabel>Head</SectionLabel>
              <InfoTip text="The last small part that turns what the base model has learned into your answer, such as a class or a number." />
            </Group>
            <Group gap="md" align="flex-end" wrap="wrap">
              <NumberInput
                size="xs"
                w={110}
                label={<LabelWithTip label="Hidden layers" tip={layersSpec.description} />}
                min={layersSpec.min ?? 0}
                max={layersSpec.max ?? 4}
                allowDecimal={false}
                clampBehavior="strict"
                value={headLayers}
                onChange={(v) => typeof v === 'number' && onValue('headLayers', v)}
              />
              {headLayers > 0 && widthSpec && (
                <Select
                  size="xs"
                  w={110}
                  label={<LabelWithTip label="Width" tip={widthSpec.description} />}
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
                  label={<LabelWithTip label="Dropout" tip={dropoutSpec.description} />}
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
                  ? 'No hidden layers: the base model feeds the output directly.'
                  : `${headLayers} × ${values.headWidth ?? widthSpec?.default ?? 256} hidden units.`}
              </Text>
            </Group>
          </Stack>
        </Paper>
      )}

      {weightsSpec && (
        <Paper p="sm">
          <Switch
            size="xs"
            label={<LabelWithTip label={weightsSpec.label} tip={weightsSpec.description} />}
            checked={values.useClassWeights === true}
            onChange={(e) => onValue('useClassWeights', e.currentTarget.checked)}
          />
          <Text size="xs" c="dimmed" mt={4}>
            Each class's loss is weighted inversely to how often it appears in the snapshot.
          </Text>
        </Paper>
      )}
    </Stack>
  )
}

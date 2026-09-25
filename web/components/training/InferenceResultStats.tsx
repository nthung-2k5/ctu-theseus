import { ActionIcon, Badge, CopyButton, Group, Paper, SimpleGrid, Slider, Stack, Text, Tooltip } from '@mantine/core'
import { CheckIcon, CopyIcon } from '@phosphor-icons/react'
import { SectionLabel } from '@public/components/ui'
import { MONO_STACK } from '@public/theme'
import { ConfidenceBars } from './ConfidenceBars'

/** Mirrors the inference output schema (ai_service/theseus/services/predict.py). */
export type InferenceOutput =
  | { kind: 'classification'; feature: string; classes: { label: string; confidence: number }[] }
  | { kind: 'regression'; feature: string; value: number }
  | { kind: 'text'; feature: string; text: string }
  | { kind: 'tokens'; feature: string; tokens: { token: string; tag: string }[] }

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

function Kpi({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <Text fw={600} size="lg" className="tnum" style={color ? { color } : undefined}>
        {value}
      </Text>
    </div>
  )
}

function confidenceColor(confidence: number) {
  return confidence > 0.8
    ? 'var(--mantine-color-teal-4)'
    : confidence > 0.5
      ? 'var(--mantine-color-yellow-4)'
      : 'var(--mantine-color-red-4)'
}

function Classification({
  output,
  threshold,
  onThresholdChange,
}: {
  output: Extract<InferenceOutput, { kind: 'classification' }>
  threshold: number
  onThresholdChange: (v: number) => void
}) {
  const ranked = [...output.classes].sort((a, b) => b.confidence - a.confidence)
  const [top, second] = ranked
  const shown = ranked.filter((c) => c.confidence >= threshold)

  if (!top) {
    return (
      <Paper p="md">
        <Text size="sm" c="dimmed" ta="center">
          The model returned no classes.
        </Text>
      </Paper>
    )
  }

  return (
    <>
      <Paper p="md">
        <SectionLabel>Prediction · {output.feature}</SectionLabel>
        <Text fw={600} size="xl" mt={2} truncate>
          {top.label}
        </Text>
        <SimpleGrid cols={3} mt="sm" spacing="sm">
          <Kpi label="Confidence" value={pct(top.confidence)} color={confidenceColor(top.confidence)} />
          <Kpi label="Margin" value={second ? pct(top.confidence - second.confidence) : '—'} />
          <Kpi label="Classes" value={String(ranked.length)} />
        </SimpleGrid>
      </Paper>

      <Paper p="md">
        <Stack gap="xs">
          <Group justify="space-between" wrap="nowrap">
            <SectionLabel>Confidence</SectionLabel>
            <Text size="xs" c="dimmed" className="tnum">
              {shown.length} of {ranked.length} ≥ {Math.round(threshold * 100)}%
            </Text>
          </Group>
          {shown.length > 0 ? (
            <ConfidenceBars classes={shown} />
          ) : (
            <Text size="sm" c="dimmed" ta="center" py="sm">
              No class reaches the confidence threshold.
            </Text>
          )}
          <div>
            <Text size="xs" c="dimmed" mb={4}>
              Confidence threshold
            </Text>
            <Slider
              size="sm"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={onThresholdChange}
              label={(v) => `${Math.round(v * 100)}%`}
            />
          </div>
        </Stack>
      </Paper>
    </>
  )
}

function Regression({ output }: { output: Extract<InferenceOutput, { kind: 'regression' }> }) {
  return (
    <Paper p="md">
      <SectionLabel>Predicted · {output.feature}</SectionLabel>
      <Text fw={600} size="2rem" className="tnum" style={{ color: 'var(--mantine-color-cyan-4)' }}>
        {Number.isInteger(output.value) ? output.value : Number(output.value.toPrecision(6))}
      </Text>
    </Paper>
  )
}

function GeneratedText({ output }: { output: Extract<InferenceOutput, { kind: 'text' }> }) {
  const words = output.text.trim() ? output.text.trim().split(/\s+/).length : 0
  return (
    <Paper p="md">
      <Stack gap="xs">
        <Group justify="space-between" wrap="nowrap">
          <SectionLabel>Output · {output.feature}</SectionLabel>
          <CopyButton value={output.text} timeout={1500}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? 'Copied' : 'Copy'}>
                <ActionIcon variant="subtle" color={copied ? 'teal' : 'gray'} onClick={copy} aria-label="Copy output">
                  {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
                </ActionIcon>
              </Tooltip>
            )}
          </CopyButton>
        </Group>
        <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
          {output.text}
        </Text>
        <SimpleGrid cols={2} spacing="sm">
          <Kpi label="Words" value={String(words)} />
          <Kpi label="Characters" value={String(output.text.length)} />
        </SimpleGrid>
      </Stack>
    </Paper>
  )
}

function Tokens({ output }: { output: Extract<InferenceOutput, { kind: 'tokens' }> }) {
  const tagged = output.tokens.filter((t) => t.tag !== 'O')
  const byTag = new Map<string, number>()
  for (const t of tagged) byTag.set(t.tag, (byTag.get(t.tag) ?? 0) + 1)
  const tags = [...byTag].sort((a, b) => b[1] - a[1])

  return (
    <>
      <Paper p="md">
        <SectionLabel mb={6}>Tagged tokens · {output.feature}</SectionLabel>
        <Group gap={6}>
          {output.tokens.map((t, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: token/tag pairs have no stable identity
            <Tooltip key={i} label={t.tag}>
              <Badge color={t.tag === 'O' ? 'gray' : 'cyan'} variant={t.tag === 'O' ? 'light' : 'filled'} size="lg">
                <span style={{ fontFamily: MONO_STACK }}>{t.token || '·'}</span>
              </Badge>
            </Tooltip>
          ))}
        </Group>
      </Paper>
      <Paper p="md">
        <SimpleGrid cols={3} spacing="sm">
          <Kpi label="Tokens" value={String(output.tokens.length)} />
          <Kpi label="Tagged" value={String(tagged.length)} />
          <Kpi label="Tags" value={String(tags.length)} />
        </SimpleGrid>
        {tags.length > 0 && (
          <Group gap={6} mt="sm">
            {tags.map(([tag, n]) => (
              <Badge key={tag} color="cyan">
                {tag} · {n}
              </Badge>
            ))}
          </Group>
        )}
      </Paper>
    </>
  )
}

/** Parses an inference result into the statistics panels of the playground's result column. */
export function InferenceResultStats({
  output,
  threshold,
  onThresholdChange,
}: {
  output: InferenceOutput
  threshold: number
  onThresholdChange: (v: number) => void
}) {
  return (
    <Stack gap="md">
      {output.kind === 'classification' && (
        <Classification output={output} threshold={threshold} onThresholdChange={onThresholdChange} />
      )}
      {output.kind === 'regression' && <Regression output={output} />}
      {output.kind === 'text' && <GeneratedText output={output} />}
      {output.kind === 'tokens' && <Tokens output={output} />}
    </Stack>
  )
}

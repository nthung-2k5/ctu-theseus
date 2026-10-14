/**
 * The footer under every staging view: how the staged batch divides into splits, and the Upload button.
 *
 * It sits at the bottom, where review ends. It reads a plain `CommitSummary`, so the file view and the inline-item
 * view feed it the same way. Problems (errors, duplicates, missing classes) are shown here only when nothing else on
 * the screen already lists them. Pressing Upload hands the batch to a background job (see `useCommit`); nothing here
 * sends anything.
 */

import { Button, Checkbox, Group, NumberInput, Paper, Popover, Progress, Stack, Text } from '@mantine/core'
import { ShuffleIcon, TrashIcon, UploadSimpleIcon } from '@phosphor-icons/react'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import type { CommitSettings } from '@public/lib/upload/commit'
import { ratioTotal, validRatios } from '@public/lib/upload/splitPlan'
import type { StagedSplit } from '@public/lib/upload/types'
import { plural } from './useCommit'

export interface CommitSummary {
  /** Everything staged, whether or not it will be sent. */
  total: number
  /** What an upload would send. */
  uploadable: number
  bySplit: Record<StagedSplit, number>
  errors: number
  warnings: number
  /** Staged with no class, for a task that labels with classes. */
  unclassified: number
  /** Left out because the dataset already has the content. */
  skipped: number
  newClasses: number
}

function AutoSplitSettings({
  settings,
  onChange,
  canStratify,
}: {
  settings: CommitSettings
  onChange: (settings: CommitSettings) => void
  canStratify: boolean
}) {
  const { ratios } = settings
  const total = ratioTotal(ratios)
  const set = (key: keyof typeof ratios) => (value: string | number) =>
    onChange({ ...settings, ratios: { ...ratios, [key]: Number(value) || 0 } })
  const percent = (n: number) => (total > 0 ? Math.round((n / total) * 100) : 0)
  return (
    <Popover width={300} position="bottom-end" withArrow shadow="md">
      <Popover.Target>
        <Button size="compact-xs" variant="subtle" leftSection={<ShuffleIcon size={12} />}>
          {percent(ratios.train)} / {percent(ratios.validation)} / {percent(ratios.test)}
        </Button>
      </Popover.Target>
      <Popover.Dropdown>
        <Stack gap="xs">
          <Text size="xs" c="dimmed">
            Files and rows with no split of their own are shuffled into train / validation / test in this ratio when you
            upload. Anything already in the dataset keeps its split.
          </Text>
          <Group grow>
            <NumberInput size="xs" label="Train" min={0} value={ratios.train} onChange={set('train')} />
            <NumberInput size="xs" label="Validation" min={0} value={ratios.validation} onChange={set('validation')} />
            <NumberInput size="xs" label="Test" min={0} value={ratios.test} onChange={set('test')} />
          </Group>
          {!validRatios(ratios) && (
            <Text size="xs" c="red.7">
              Give at least one split a share.
            </Text>
          )}
          {canStratify && (
            <Checkbox
              size="xs"
              label="Keep every class in every split"
              description="Each class is split on its own, so a class cannot end up entirely in train."
              checked={settings.stratify}
              onChange={(event) => onChange({ ...settings, stratify: event.currentTarget.checked })}
            />
          )}
        </Stack>
      </Popover.Dropdown>
    </Popover>
  )
}

export function CommitBar({
  summary,
  settings,
  onSettingsChange,
  canStratify,
  checking,
  starting,
  onUpload,
  onDiscard,
  locked = false,
  showProblems = true,
}: {
  summary: CommitSummary
  settings: CommitSettings
  onSettingsChange: (settings: CommitSettings) => void
  canStratify: boolean
  /** The background duplicate check, while it runs. */
  checking?: { label: string; fraction: number } | null
  starting: boolean
  onUpload: () => void
  onDiscard: () => void
  /** Something is being read or added, so the batch is not final yet. */
  locked?: boolean
  /** List errors, duplicates and missing classes here too (false where a summary already does). */
  showProblems?: boolean
}) {
  const { bySplit } = summary
  const needsRatios = bySplit.auto > 0
  const ratiosOk = !needsRatios || validRatios(settings.ratios)
  // A batch of nothing but already-added files still has something to do: take them off the list.
  const canUpload = (summary.uploadable > 0 || summary.skipped > 0) && ratiosOk && !locked
  const label =
    summary.uploadable > 0
      ? `Upload ${plural(summary.uploadable, 'file')}`
      : summary.skipped > 0
        ? 'Clear already-added files'
        : 'Upload'
  // Why the button is greyed out, so a disabled Upload is never a mystery.
  const why =
    summary.total === 0 ? 'Add something to upload first' : !ratiosOk ? 'Give at least one split a share' : null

  return (
    <Paper withBorder p="xs" radius="md" style={{ flexShrink: 0 }}>
      <Group justify="space-between" wrap="nowrap" gap="md">
        <Stack gap={2} style={{ minWidth: 0 }}>
          <Group gap="md" wrap="wrap">
            <Text size="sm" fw={600}>
              {summary.total === 0 ? 'Nothing staged' : `${summary.total.toLocaleString()} staged`}
            </Text>
            {summary.total > 0 &&
              SPLIT_OPTIONS.map(({ value, label: splitLabel }) => (
                <Text key={value} size="xs" c="dimmed">
                  {splitLabel}{' '}
                  <Text component="span" fw={600} c="var(--mantine-color-text)">
                    {bySplit[value as keyof typeof bySplit].toLocaleString()}
                  </Text>
                </Text>
              ))}
            {needsRatios && (
              <Group gap={2} wrap="nowrap">
                <Text size="xs" c="dimmed">
                  Auto-split{' '}
                  <Text component="span" fw={600} c="var(--mantine-color-text)">
                    {bySplit.auto.toLocaleString()}
                  </Text>
                </Text>
                <AutoSplitSettings settings={settings} onChange={onSettingsChange} canStratify={canStratify} />
              </Group>
            )}
          </Group>
          {showProblems && summary.total > 0 && (
            <Group gap="md" wrap="wrap">
              {summary.errors > 0 && (
                <Text size="xs" c="red.7">
                  {plural(summary.errors, 'file')} with errors won't be uploaded
                </Text>
              )}
              {summary.skipped > 0 && (
                <Text size="xs" c="dimmed">
                  {summary.skipped.toLocaleString()} already in the dataset won't be sent
                </Text>
              )}
              {summary.warnings > 0 && (
                <Text size="xs" c="yellow.7">
                  {plural(summary.warnings, 'file')} with warnings
                </Text>
              )}
              {summary.unclassified > 0 && (
                <Text size="xs" c="yellow.7">
                  {summary.unclassified.toLocaleString()} without a class
                </Text>
              )}
              {summary.newClasses > 0 && (
                <Text size="xs" c="teal.7">
                  {plural(summary.newClasses, 'new class')} will be created
                </Text>
              )}
            </Group>
          )}
        </Stack>
        <Group gap="xs" wrap="nowrap" style={{ flexShrink: 0 }}>
          {checking && (
            <Group gap={6} wrap="nowrap">
              <Progress value={checking.fraction * 100} w={90} size="sm" />
              <Text size="xs" c="dimmed" visibleFrom="md">
                {checking.label}…
              </Text>
            </Group>
          )}
          {summary.total > 0 && (
            <Button
              size="xs"
              variant="subtle"
              color="red"
              leftSection={<TrashIcon size={14} />}
              onClick={onDiscard}
              disabled={starting}
            >
              Discard
            </Button>
          )}
          {why && !canUpload && (
            <Text size="xs" c="dimmed" visibleFrom="sm">
              {why}
            </Text>
          )}
          <Button
            leftSection={<UploadSimpleIcon size={14} />}
            onClick={onUpload}
            loading={starting}
            disabled={!canUpload}
          >
            {label}
          </Button>
        </Group>
      </Group>
    </Paper>
  )
}

/**
 * The card at the top of a file import: what was dropped, in one glance.
 *
 * Folder layouts are guessed (`lib/upload/layout.ts`), so the guess is stated rather than silently applied: how many
 * files, how they divide between classes (a balance bar, so a lopsided or tiny class stands out), where each file's
 * split came from, and what is wrong with some of them. The handles to correct a reading live here — re-read the
 * layout, give the files with no split a fixed one, drop the broken ones. Fixing a class is done in the lanes, the
 * folders or the table, where the files are.
 */

import { Button, Group, Menu, Paper, Select, Stack, Text } from '@mantine/core'
import { CaretDownIcon, TrashIcon } from '@phosphor-icons/react'
import { ProportionBar } from '@public/components/ui'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import type { CommitSettings } from '@public/lib/upload/commit'
import { hasBlockingError } from '@public/lib/upload/fileTree'
import { LAYOUT_LABELS } from '@public/lib/upload/layout'
import { ratioTotal } from '@public/lib/upload/splitPlan'
import type { LayoutKind, StagedSplit } from '@public/lib/upload/types'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import { laneColor } from './laneColor'
import { plural } from './useCommit'

const MAX_LEGEND = 8

export function ImportSummary({
  staging,
  taskUsesClasses,
  settings,
  locked,
}: {
  staging: FileStaging
  taskUsesClasses: boolean
  settings: CommitSettings
  locked: boolean
}) {
  const { index, state, relayout, move, remove } = staging
  const { totals } = index
  if (totals.files === 0) return null

  const batch = state.lastBatch
  // Classes with files in them, largest first: the balance bar and its legend read from the biggest down.
  const populated = index.folders.filter((f) => !f.isNone && f.total > 0).sort((a, b) => b.total - a.total)
  const files = [...state.files.values()]
  const autoCount = totals.bySplit.auto
  const placed = totals.files - autoCount
  const ratioSum = ratioTotal(settings.ratios)
  const ratioText =
    ratioSum > 0
      ? [settings.ratios.train, settings.ratios.validation, settings.ratios.test]
          .map((n) => Math.round((n / ratioSum) * 100))
          .join(' / ')
      : null
  const splitText =
    autoCount > 0
      ? `${placed > 0 ? `${placed.toLocaleString()} placed by folder, ` : ''}${autoCount.toLocaleString()} will be auto-split${
          ratioText ? ` ${ratioText}` : ''
        }`
      : `${placed.toLocaleString()} placed by folder`

  return (
    <Paper withBorder p="sm" style={{ flexShrink: 0 }}>
      <Stack gap={6}>
        <Group justify="space-between" wrap="nowrap" align="flex-start">
          <Group gap="sm" align="baseline" wrap="wrap">
            <Text fw={700} size="lg">
              {plural(totals.files, 'file')}
            </Text>
            {taskUsesClasses && (
              <Text size="sm" c="dimmed">
                {populated.length === 0
                  ? 'no class folders found'
                  : `${plural(populated.length, 'class')}${
                      totals.newClassNames.length > 0 ? `, ${totals.newClassNames.length} new` : ''
                    }`}
              </Text>
            )}
          </Group>
          {batch && batch.candidates.length > 1 && (
            <Group gap={6} wrap="nowrap">
              <Text size="xs" c="dimmed">
                Folders read as
              </Text>
              <Select
                size="xs"
                w={160}
                aria-label="How the last drop's folders are read"
                data={batch.candidates.map((kind) => ({ value: kind, label: LAYOUT_LABELS[kind] }))}
                value={batch.kind}
                onChange={(kind) => kind && relayout(kind as LayoutKind)}
                allowDeselect={false}
                disabled={locked}
              />
            </Group>
          )}
        </Group>

        {taskUsesClasses && populated.length > 0 && (
          <Stack gap={4}>
            <ProportionBar
              height={10}
              segments={populated.map((f) => ({
                key: f.key,
                label: f.name,
                value: f.total,
                color: laneColor(f),
              }))}
            />
            <Group gap="md" wrap="wrap">
              {populated.slice(0, MAX_LEGEND).map((f) => (
                <Group key={f.key} gap={4} wrap="nowrap">
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: laneColor(f), flexShrink: 0 }} />
                  <Text size="xs">
                    {f.name}{' '}
                    <Text component="span" size="xs" c="dimmed">
                      {f.total.toLocaleString()}
                      {f.pending ? ' · new' : ''}
                    </Text>
                  </Text>
                </Group>
              ))}
              {populated.length > MAX_LEGEND && (
                <Text size="xs" c="dimmed">
                  +{populated.length - MAX_LEGEND} more
                </Text>
              )}
            </Group>
          </Stack>
        )}

        <Group gap="md" wrap="wrap">
          <Group gap={2} wrap="nowrap">
            <Text size="xs" c="dimmed">
              {splitText}
            </Text>
            {autoCount > 0 && !locked && (
              <Menu position="bottom-start" withinPortal>
                <Menu.Target>
                  <Button size="compact-xs" variant="subtle" rightSection={<CaretDownIcon size={10} />}>
                    put them in…
                  </Button>
                </Menu.Target>
                <Menu.Dropdown>
                  {SPLIT_OPTIONS.map((option) => (
                    <Menu.Item
                      key={option.value}
                      onClick={() =>
                        move(
                          files.filter((f) => f.split === 'auto').map((f) => f.id),
                          { split: option.value as StagedSplit },
                        )
                      }
                    >
                      {option.label}
                    </Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>
            )}
          </Group>
          {totals.withErrors > 0 && (
            <Group gap={2} wrap="nowrap">
              <Text size="xs" c="red.7">
                {plural(totals.withErrors, 'file')} can't be uploaded
              </Text>
              {!locked && (
                <Button
                  size="compact-xs"
                  variant="subtle"
                  color="red"
                  leftSection={<TrashIcon size={12} />}
                  onClick={() => remove(files.filter(hasBlockingError).map((f) => f.id))}
                >
                  remove
                </Button>
              )}
            </Group>
          )}
          {totals.skipped > 0 && (
            <Text size="xs" c="dimmed">
              {totals.skipped.toLocaleString()} already in the dataset (or repeated) won't be sent
            </Text>
          )}
          {taskUsesClasses && totals.unclassified > 0 && (
            <Text size="xs" c="yellow.7">
              {totals.unclassified.toLocaleString()} without a class
            </Text>
          )}
        </Group>
      </Stack>
    </Paper>
  )
}

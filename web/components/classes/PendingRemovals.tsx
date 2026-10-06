import { Button, Collapse, Group, Paper, Stack, Text, UnstyledButton } from '@mantine/core'
import { ArrowUUpLeftIcon, CaretDownIcon, CaretRightIcon } from '@phosphor-icons/react'
import type { DraftClass } from '@public/lib/classDraft'
import { useState } from 'react'

/**
 * Classes this save would remove, each with what happens to its items, and a way back. It makes a removal
 * recoverable without throwing away every other edit (Discard), and says plainly where the items go.
 */
export function PendingRemovals({
  removed,
  countOf,
  destinationOf,
  onRestore,
}: {
  removed: DraftClass[]
  countOf: (row: DraftClass) => number | undefined
  /** The name of the class a removed class's items move to, or undefined when they are left unlabelled. */
  destinationOf: (row: DraftClass) => string | undefined
  onRestore: (row: DraftClass) => void
}) {
  const [open, setOpen] = useState(true)
  if (removed.length === 0) return null

  return (
    <Paper p="sm" withBorder>
      <UnstyledButton onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <Group gap={6}>
          {open ? <CaretDownIcon size={12} /> : <CaretRightIcon size={12} />}
          <Text size="sm" fw={600}>
            Removed when you save ({removed.length})
          </Text>
        </Group>
      </UnstyledButton>
      <Collapse expanded={open}>
        <Stack gap={6} mt="xs">
          {removed.map((row) => {
            const n = countOf(row)
            const to = destinationOf(row)
            return (
              <Group key={row.key} justify="space-between" wrap="nowrap">
                <Text size="sm" style={{ minWidth: 0 }} truncate>
                  {row.name}{' '}
                  <Text span size="xs" c="dimmed">
                    {n === undefined || n === 0
                      ? ''
                      : to
                        ? `· ${n.toLocaleString()} items move to "${to}"`
                        : `· ${n.toLocaleString()} items become unlabelled`}
                  </Text>
                </Text>
                <Button
                  size="compact-xs"
                  variant="subtle"
                  leftSection={<ArrowUUpLeftIcon size={12} />}
                  onClick={() => onRestore(row)}
                  aria-label={`Restore ${row.name}`}
                >
                  Restore
                </Button>
              </Group>
            )
          })}
        </Stack>
      </Collapse>
    </Paper>
  )
}

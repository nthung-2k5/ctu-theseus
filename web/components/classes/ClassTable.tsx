import { ColorInput, Group, Table, Text, TextInput } from '@mantine/core'
import {
  attention,
  CLASS_PALETTE,
  type ClassCounts,
  type DraftClass,
  nameError,
  norm,
  type SplitName,
} from '@public/lib/classDraft'
import type { ReactNode } from 'react'
import { AttentionBadge } from './AttentionBadge'
import { SplitMeter } from './SplitMeter'

/**
 * The editable class list. `rows` is what is shown (filtered and sorted); `allRows` is the whole draft, which
 * name uniqueness is checked against so hiding a row never hides a clash.
 *
 * The header sticks under the app header while the page scrolls, so there is deliberately no horizontal scroll
 * container here (it would become the sticky parent and break that). Narrow screens drop the description
 * column and show the description under the name instead.
 */
export function ClassTable({
  rows,
  allRows,
  stats,
  max,
  min,
  splitsInUse,
  showColor,
  onPatch,
  renderMenu,
  emptyMessage,
}: {
  rows: DraftClass[]
  allRows: DraftClass[]
  /** Per-class counts by class id; undefined until the stats have loaded. */
  stats: Map<string, ClassCounts> | undefined
  /** The largest class's item count, which the count bars are scaled to. */
  max: number
  min: number
  splitsInUse: ReadonlySet<SplitName>
  /** Only tasks that draw their classes on the item (boxes, masks) have a use for a colour. */
  showColor: boolean
  onPatch: (key: string, change: Partial<DraftClass>) => void
  renderMenu: (row: DraftClass) => ReactNode
  emptyMessage: string
}) {
  return (
    <Table withTableBorder verticalSpacing={6} highlightOnHover stickyHeader stickyHeaderOffset={48}>
      <Table.Thead>
        <Table.Tr>
          {showColor && <Table.Th w={140}>Colour</Table.Th>}
          <Table.Th w={230}>Name</Table.Th>
          <Table.Th visibleFrom="md">Description</Table.Th>
          <Table.Th w={300}>Items in draft</Table.Th>
          <Table.Th w={48}>
            <span style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0 0 0 0)' }}>
              Actions
            </span>
          </Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {rows.map((row) => {
          const error = nameError(row.name, allRows, row.key)
          const counts = row.classId ? stats?.get(row.classId) : undefined
          return (
            <Table.Tr key={row.key}>
              {showColor && (
                <Table.Td>
                  <ColorInput
                    size="xs"
                    value={row.color}
                    format="hex"
                    swatches={CLASS_PALETTE}
                    withEyeDropper={false}
                    popoverProps={{ withinPortal: true }}
                    aria-label={`Colour of ${row.name}`}
                    onChange={(v) => onPatch(row.key, { color: v })}
                  />
                </Table.Td>
              )}
              <Table.Td>
                <TextInput
                  size="xs"
                  value={row.name}
                  error={error}
                  aria-label={`Name of ${row.name || 'new class'}`}
                  onChange={(e) => onPatch(row.key, { name: e.currentTarget.value })}
                  onBlur={() => !error && onPatch(row.key, { name: norm(row.name) })}
                />
                <TextInput
                  hiddenFrom="md"
                  mt={4}
                  size="xs"
                  value={row.description}
                  placeholder="Description (optional)"
                  aria-label={`Description of ${row.name}`}
                  onChange={(e) => onPatch(row.key, { description: e.currentTarget.value })}
                />
              </Table.Td>
              <Table.Td visibleFrom="md">
                <TextInput
                  size="xs"
                  value={row.description}
                  placeholder="Description (optional)"
                  aria-label={`Description of ${row.name}`}
                  onChange={(e) => onPatch(row.key, { description: e.currentTarget.value })}
                />
              </Table.Td>
              <Table.Td>
                {!row.classId ? (
                  <Text size="xs" c="dimmed">
                    new
                  </Text>
                ) : !counts ? (
                  <Text size="xs" c="dimmed">
                    —
                  </Text>
                ) : (
                  <Group gap="xs" wrap="nowrap">
                    <div style={{ flex: 1, minWidth: 60 }}>
                      <SplitMeter counts={counts} max={max} />
                    </div>
                    <Text size="xs" className="tnum" w={48} ta="right">
                      {counts.total.toLocaleString()}
                    </Text>
                    <AttentionBadge attention={attention(counts, min, splitsInUse)} min={min} />
                  </Group>
                )}
              </Table.Td>
              <Table.Td>{renderMenu(row)}</Table.Td>
            </Table.Tr>
          )
        })}
        {rows.length === 0 && (
          <Table.Tr>
            <Table.Td colSpan={showColor ? 5 : 4}>
              <Text ta="center" c="dimmed" py="md">
                {emptyMessage}
              </Text>
            </Table.Td>
          </Table.Tr>
        )}
      </Table.Tbody>
    </Table>
  )
}

import { Group, Pagination, Paper, Select, Text } from '@mantine/core'

export const PER_PAGE_OPTIONS = [20, 50, 100, 200, 500, 1000] as const

export interface ItemsPaginationBarProps {
  total: number
  page: number
  perPage: number
  onPageChange: (page: number) => void
  onPerPageChange: (perPage: number) => void
}

/**
 * Total count, per-page size, and page controls for an item list. It doesn't
 * stick to the viewport itself: put it in an `ItemsFooter`, which keeps it (and
 * any bulk-action bar above it) on screen while the list scrolls past. Shared
 * by the Dataset draft view and the Snapshot detail view — both browse the same
 * paginated item list shape.
 */
export function ItemsPaginationBar({ total, page, perPage, onPageChange, onPerPageChange }: ItemsPaginationBarProps) {
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  return (
    <Paper p="xs" withBorder radius="md">
      <Group justify="space-between" wrap="wrap" gap="sm">
        <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
          {total} items
        </Text>

        <Group gap="sm" wrap="wrap">
          <Select
            size="sm"
            w="8rem"
            data={PER_PAGE_OPTIONS.map((n) => ({ value: String(n), label: `${n} / page` }))}
            value={String(perPage)}
            onChange={(v) => v && onPerPageChange(Number(v))}
            allowDeselect={false}
          />

          <Pagination total={totalPages} value={page} onChange={onPageChange} size="sm" />
        </Group>
      </Group>
    </Paper>
  )
}

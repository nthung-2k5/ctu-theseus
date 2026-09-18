import { Group, Pagination, Paper, Select, Text } from '@mantine/core'

export const PER_PAGE_OPTIONS = [20, 50, 100, 200, 500, 1000] as const

/**
 * Bottom toolbar for an item list: total count, per-page size, and page
 * controls. Attached to the end of the list (not a floating overlay), but
 * sticky so it stays on screen at the bottom of the viewport while the list
 * scrolls past it. Shared by the Dataset draft view and the Snapshot detail
 * view — both browse the same paginated item list shape.
 */
export function ItemsPaginationBar({
  total,
  page,
  perPage,
  onPageChange,
  onPerPageChange,
}: {
  total: number
  page: number
  perPage: number
  onPageChange: (page: number) => void
  onPerPageChange: (perPage: number) => void
}) {
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  return (
    <Paper p="xs" bdrs={0} mx={'-1.25rem'} pos="sticky" bottom={0} withBorder style={{ width: 'calc(100% + 2.5rem)', borderWidth: '1px 0 0 0' }}>
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

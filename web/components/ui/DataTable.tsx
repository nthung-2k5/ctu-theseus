import { ScrollArea, Table } from '@mantine/core'
import type { ReactNode } from 'react'
import { EmptyState } from './EmptyState'

export interface DataTableColumn<T> {
  key: string
  header: ReactNode
  fit?: boolean
  render: (row: T) => ReactNode
}

/** Loading/empty/render-rows boilerplate shared by every table in the app (pool items, split items, ...). */
export function DataTable<T>({
  columns,
  data,
  getRowKey,
  loading,
  emptyMessage = 'No items yet',
  onRowClick,
}: {
  columns: DataTableColumn<T>[]
  data: T[]
  getRowKey: (row: T) => string
  loading?: boolean
  emptyMessage?: ReactNode
  onRowClick?: (row: T) => void
}) {
  if (loading) return <EmptyState loading compact />
  if (data.length === 0) return <EmptyState description={emptyMessage} compact />

  return (
    <ScrollArea>
      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            {columns.map((c) => (
              <Table.Th key={c.key} style={{ width: c.fit ? 'fit-content' : undefined }}>
                {c.header}
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {data.map((row) => (
            <Table.Tr
              key={getRowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              style={onRowClick ? { cursor: 'pointer' } : undefined}
            >
              {columns.map((c) => (
                <Table.Td key={c.key} style={{ width: c.fit ? 'fit-content' : undefined }}>
                  {c.render(row)}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  )
}

import { Box } from '@mantine/core'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type ReactNode, useRef } from 'react'
import { EmptyState } from './EmptyState'

export interface VirtualColumn<T> {
  key: string
  header: ReactNode
  /** Fixed width in px. Columns without one share the remaining space. */
  width?: number
  render: (row: T) => ReactNode
}

const GAP = 12

/**
 * `DataTable` for lists that can run to thousands of rows: only the rows in (or near) the viewport are
 * mounted. It fills its parent's height and scrolls itself, and rows have a fixed height, so columns take
 * explicit widths rather than sizing to content.
 */
export function VirtualDataTable<T>({
  columns,
  data,
  getRowKey,
  rowHeight = 44,
  emptyMessage = 'No items yet',
}: {
  columns: VirtualColumn<T>[]
  data: T[]
  getRowKey: (row: T) => string
  rowHeight?: number
  emptyMessage?: ReactNode
}) {
  // TanStack Virtual returns functions the React Compiler would otherwise memoize away.
  'use no memo'

  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: data.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => rowHeight,
    overscan: 8,
    getItemKey: (i) => getRowKey(data[i]),
  })

  if (data.length === 0) return <EmptyState description={emptyMessage} compact />

  const template = columns.map((c) => (c.width ? `${c.width}px` : 'minmax(0, 1fr)')).join(' ')
  const grid = {
    display: 'grid',
    gridTemplateColumns: template,
    columnGap: GAP,
    alignItems: 'center',
    padding: '0 8px',
  }

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <Box
        style={{ ...grid, height: 36, borderBottom: '1px solid var(--mantine-color-default-border)', fontWeight: 600 }}
        fz="xs"
        c="dimmed"
      >
        {columns.map((c) => (
          <div key={c.key}>{c.header}</div>
        ))}
      </Box>
      <div ref={scrollRef} style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
          {virtualizer.getVirtualItems().map((item) => (
            <div
              key={item.key}
              style={{
                ...grid,
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: item.size,
                transform: `translateY(${item.start}px)`,
                borderBottom: '1px solid var(--mantine-color-default-border)',
              }}
            >
              {columns.map((c) => (
                <div key={c.key} style={{ minWidth: 0 }}>
                  {c.render(data[item.index])}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </Box>
  )
}

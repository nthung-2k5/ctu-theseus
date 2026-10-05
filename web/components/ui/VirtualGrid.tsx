import { useElementSize, useMergedRef } from '@mantine/hooks'
import { columnsFor, itemsInRow, rowCount } from '@public/lib/grid'
import { useWindowVirtualizer } from '@tanstack/react-virtual'
import { type ReactNode, useEffect, useRef } from 'react'

/**
 * A responsive card grid that mounts only the rows near the viewport, for pages of hundreds of items.
 * It scrolls with the page (the window), not inside a box of its own, so it drops into a normal page flow;
 * rows are measured, so cards may differ in height. Columns are as many `minItemWidth`-wide cards as fit.
 */
export function VirtualGrid<T>({
  items,
  getKey,
  renderItem,
  minItemWidth = 180,
  gap = 12,
  estimateRowHeight = 190,
}: {
  items: T[]
  getKey: (item: T) => string
  renderItem: (item: T) => ReactNode
  minItemWidth?: number
  gap?: number
  /** Only the first paint uses this; real heights are measured. */
  estimateRowHeight?: number
}) {
  // TanStack Virtual returns functions the React Compiler would otherwise memoize away.
  'use no memo'

  const { ref: sizeRef, width } = useElementSize()
  const containerRef = useRef<HTMLDivElement>(null)
  const ref = useMergedRef(sizeRef, containerRef)

  const columns = columnsFor(width, minItemWidth, gap)
  const virtualizer = useWindowVirtualizer({
    count: rowCount(items.length, columns),
    estimateSize: () => estimateRowHeight + gap,
    overscan: 3,
    // Where the grid starts on the page, so row offsets are relative to it rather than to the document.
    scrollMargin: containerRef.current ? containerRef.current.getBoundingClientRect().top + window.scrollY : 0,
  })

  // Cards re-flow when the column count changes, so every cached row height is stale.
  const { measure } = virtualizer
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure rows only when the column count changes
  useEffect(() => measure(), [columns])

  return (
    <div ref={ref} style={{ position: 'relative', height: virtualizer.getTotalSize(), width: '100%' }}>
      {virtualizer.getVirtualItems().map((row) => (
        <div
          key={row.key}
          data-index={row.index}
          ref={virtualizer.measureElement}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            transform: `translateY(${row.start - virtualizer.options.scrollMargin}px)`,
            display: 'grid',
            gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
            columnGap: gap,
            paddingBottom: gap,
          }}
        >
          {itemsInRow(items, columns, row.index).map((item) => (
            // A one-cell grid, so the card stretches to the row's height like a direct grid child would.
            <div key={getKey(item)} style={{ display: 'grid', minWidth: 0 }}>
              {renderItem(item)}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

/** Layout arithmetic for `components/ui/VirtualGrid`, kept pure so it can be unit-tested. */

/** How many columns of at least `minItemWidth` fit in `width`, with `gap` between them. Always at least one. */
export const columnsFor = (width: number, minItemWidth: number, gap: number): number =>
  width <= 0 ? 1 : Math.max(1, Math.floor((width + gap) / (minItemWidth + gap)))

export const rowCount = (itemCount: number, columns: number): number => Math.ceil(itemCount / columns)

/** The slice of `items` that row `row` shows. */
export const itemsInRow = <T>(items: readonly T[], columns: number, row: number): T[] =>
  items.slice(row * columns, (row + 1) * columns)

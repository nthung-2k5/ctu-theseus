/** "image_horizontal_flip" -> "Horizontal flip": the op ids are `<modality>_<name>`. */
export const opLabel = (id: string) => {
  const words = id.split('_').slice(1).join(' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** The ops recorded on a preprocessed or augmented item's `{ops: [{id, params}]}` (or `{copy, ops}`) field. */
export const appliedOps = (record: unknown): string[] => {
  const ops = (record as { ops?: { id: string }[] } | null | undefined)?.ops
  return Array.isArray(ops) ? ops.map((o) => opLabel(o.id)) : []
}

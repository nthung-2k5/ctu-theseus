/**
 * The runner for inline items (text and tabular rows): sends a job's items through `POST /projects/:id/items`.
 *
 * Items go in chunks, so a big import is not one enormous request and one rejected chunk costs only its own items.
 * The server answers with the rows it created and the indexes of the ones it refused.
 */

import type { ItemIn } from '../api/generated/models/itemIn'
import type { JobRunContext, JobUnit } from './jobs'
import { describeError } from './uploadStaged'

export const ITEMS_PER_REQUEST = 500

/** An item as the server takes it, minus the split (a unit carries that, already decided). */
export type InlinePayload = Omit<ItemIn, 'split'>

export type InlineUnit = JobUnit<InlinePayload>

export interface InlineApi {
  create: (
    projectId: string,
    items: ItemIn[],
    signal: AbortSignal,
  ) => Promise<{ failed: { index: number; message: string }[] }>
}

export function inlineRunner(options: {
  api: InlineApi
  projectId: string
}): (units: InlineUnit[], ctx: JobRunContext<InlinePayload>) => Promise<void> {
  const { api, projectId } = options

  return async (units, ctx) => {
    for (let start = 0; start < units.length && !ctx.signal.aborted; start += ITEMS_PER_REQUEST) {
      const chunk = units.slice(start, start + ITEMS_PER_REQUEST)
      try {
        const { failed } = await api.create(
          projectId,
          chunk.map((unit) => ({ split: unit.split, ...unit.payload })),
          ctx.signal,
        )
        const refused = new Map(failed.map((f) => [f.index, f.message]))
        ctx.settle({
          ok: chunk.filter((_, i) => !refused.has(i)),
          failed: chunk.flatMap((unit, i) => {
            const reason = refused.get(i)
            return reason === undefined ? [] : [{ unit, reason }]
          }),
        })
      } catch (error) {
        if (ctx.signal.aborted) return
        const reason = describeError(error, 'Upload request failed')
        ctx.settle({ failed: chunk.map((unit) => ({ unit, reason })) })
      }
    }
  }
}

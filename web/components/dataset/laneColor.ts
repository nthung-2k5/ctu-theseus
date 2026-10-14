import { classColor } from '@public/lib/boxes'

/** One colour per class lane, the same wherever the class appears: the lane's class id (or, for a new class, its name). */
export const laneColor = (lane: { key: string; isNone: boolean }): string =>
  lane.isNone ? classColor(null) : classColor(lane.key.startsWith('c:') ? lane.key.slice(2) : lane.key)

import { mergeQueryKeys } from '@lukemorales/query-key-factory'
import { classes } from './classes'
import { projects } from './project'
import { training } from './training'

export const queries = mergeQueryKeys(projects, training, classes)

export * from './classes'
export * from './dataset'
export * from './project'
export * from './training'
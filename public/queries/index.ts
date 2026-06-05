import { mergeQueryKeys } from '@lukemorales/query-key-factory'
import { projects } from './project'
import { training } from './training'

export const queries = mergeQueryKeys(projects, training)

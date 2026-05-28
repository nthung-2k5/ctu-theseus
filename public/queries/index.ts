import { mergeQueryKeys } from '@lukemorales/query-key-factory'
import { projects } from './project'

export const queries = mergeQueryKeys(projects)

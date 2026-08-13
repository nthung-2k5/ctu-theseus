import { treaty } from '@elysia/eden'
import type { App } from '@server'

export const api = treaty<App>('/', {
  fetch: {
    credentials: 'include',
  },
}).api

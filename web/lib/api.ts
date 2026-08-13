import { treaty } from '@elysia/eden'
import type { App } from '@server'

export const api = treaty<App>('http://localhost:3000', {
  fetch: {
    credentials: 'include',
  },
}).api

import { treaty } from '@elysia/eden'
import type { App } from '@server'

export const api = treaty<App>(window.location.origin, {
  headers: {
    credentials: 'include',
  },
}).api

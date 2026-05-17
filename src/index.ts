import { staticPlugin } from '@elysia/static'
import index from '@public/index.html'
import { Elysia } from 'elysia'

const app = new Elysia()
  .use(
    await staticPlugin({
      prefix: '/',
      bunFullstack: true,
      alwaysStatic: true,
    }),
  )
  .get('/*', index)
  .listen(3000)

console.log(`🦊 Elysia is running at http://${app.server?.hostname}:${app.server?.port}`)

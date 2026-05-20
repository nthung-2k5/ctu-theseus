import { staticPlugin } from '@elysia/static'
import { Elysia } from 'elysia'
import { auth } from './auth'

const app = new Elysia()
  .onError(({ error }) => {
    console.error(error)
    return 'Internal Server Error'
  })
  .use(
    await staticPlugin({
      prefix: '/',
      bunFullstack: true,
      alwaysStatic: true,
    }),
  )
  /* ── Authentication ── */
  .mount(auth.handler)
  // .get('/*', index)
  .listen(3000)

console.log(`CTU Theseus server is running at http://${app.server?.hostname}:${app.server?.port}`)

export type App = typeof app

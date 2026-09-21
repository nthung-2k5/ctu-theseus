import { defineConfig } from 'orval'

export default defineConfig({
  theseus: {
    input: { target: '../schema/openapi.json' },
    output: {
      mode: 'tags-split',
      target: './lib/api/generated/endpoints.ts',
      schemas: './lib/api/generated/models',
      client: 'react-query',
      httpClient: 'axios',
      clean: true,
      override: {
        mutator: { path: './lib/api/client.ts', name: 'customInstance' },
      },
    },
  },
})

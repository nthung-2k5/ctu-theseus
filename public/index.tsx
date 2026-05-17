import { Button, MantineProvider, Title } from '@mantine/core'
import { useState } from 'react'
import { createRoot } from 'react-dom/client'

import '@public/global.css'

function App() {
  const [count, setCount] = useState(0)
  const increase = () => setCount((c) => c + 1)

  return (
    <main>
      <Title order={2}>{count}</Title>
      <Button type="button" onClick={increase}>
        Increase
      </Button>
    </main>
  )
}

const root = createRoot(document.getElementById('root')!)
root.render(
  <MantineProvider>
    <App />
  </MantineProvider>,
)

/**
 * Admin > Models: the platform's model catalog. "Imported" holds the models people brought: shared ones an
 * admin imports for everyone, and the private ones users imported for themselves (which an admin can review,
 * disable and delete). "Base" switches the models that ship with a backend on and off, per task.
 */

import { Tabs } from '@mantine/core'
import { PluginTable } from '@public/components/admin/PluginTable'
import { ImportedModelsSection } from '@public/components/models/ImportedModelsSection'
import { PageHeader } from '@public/components/ui'
import { adminModelsApi } from '@public/lib/importedModels'
import { useState } from 'react'

export function AdminModelsPage() {
  const [tab, setTab] = useState<string>('custom')

  return (
    <div className="flex flex-col gap-2">
      <PageHeader
        title="Models"
        description="Which base models users can fine-tune. A model has to be Ready, enabled, and offered for the task."
      />
      <Tabs value={tab} onChange={(v) => v && setTab(v)}>
        <Tabs.List>
          <Tabs.Tab value="custom">Imported models</Tabs.Tab>
          <Tabs.Tab value="builtin">Base models</Tabs.Tab>
        </Tabs.List>
      </Tabs>
      {tab === 'custom' ? (
        <ImportedModelsSection api={adminModelsApi} />
      ) : (
        <PluginTable
          kind="builtin_model"
          hint="Base models shipped with a backend. Turn one off to stop it being chosen."
        />
      )}
    </div>
  )
}

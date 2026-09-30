/**
 * Admin > Models: the platform's model catalog. "Custom" holds the bring-your-own models: global ones an
 * admin adds for everyone, and the private ones users added for themselves (which an admin can review,
 * disable and delete). "Built-in" switches the models that ship with a backend on and off, per task.
 */

import { Tabs } from '@mantine/core'
import { PluginTable } from '@public/components/admin/PluginTable'
import { CustomModelsSection } from '@public/components/models/CustomModelsSection'
import { PageHeader } from '@public/components/ui'
import { adminModelsApi } from '@public/lib/customModels'
import { useState } from 'react'

export function AdminModelsPage() {
  const [tab, setTab] = useState<string>('custom')

  return (
    <div className="flex flex-col gap-2">
      <PageHeader
        title="Models"
        description="Which models users can train on. A model has to be Ready, enabled, and offered for the task."
      />
      <Tabs value={tab} onChange={(v) => v && setTab(v)}>
        <Tabs.List>
          <Tabs.Tab value="custom">Custom models</Tabs.Tab>
          <Tabs.Tab value="builtin">Built-in models</Tabs.Tab>
        </Tabs.List>
      </Tabs>
      {tab === 'custom' ? (
        <CustomModelsSection api={adminModelsApi} />
      ) : (
        <PluginTable
          kind="builtin_model"
          hint="Backbones shipped with a backend. Turn one off to stop it being chosen."
        />
      )}
    </div>
  )
}

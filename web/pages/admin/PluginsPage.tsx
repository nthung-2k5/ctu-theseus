/**
 * Admin > Plugins: switch trainer backends, built-in models, export formats, preprocessing and
 * augmentation ops, and whole tasks on or off, globally or for one task. The table itself is
 * `PluginTable`; this page is the tabs around it.
 */

import { Tabs } from '@mantine/core'
import { PluginTable } from '@public/components/admin/PluginTable'
import { PageHeader } from '@public/components/ui'
import { useState } from 'react'

const KINDS = [
  { value: 'backend', label: 'Trainer backends', hint: 'Which frameworks can start new runs.' },
  { value: 'builtin_model', label: 'Built-in models', hint: 'Backbones shipped with a backend.' },
  { value: 'export_format', label: 'Export formats', hint: 'What a finished run can be exported as.' },
  { value: 'preprocessing', label: 'Preprocessing', hint: 'Snapshot-time preprocessing ops.' },
  { value: 'augmentation', label: 'Augmentation', hint: 'Snapshot-time augmentation ops.' },
  { value: 'task', label: 'Tasks', hint: 'Which tasks new projects can be created for.' },
] as const

export function PluginsPage() {
  const [kind, setKind] = useState<string>('backend')
  const current = KINDS.find((k) => k.value === kind)

  return (
    <div className="flex flex-col gap-2">
      <PageHeader
        title="Plugins & availability"
        description="Turn things off to stop new use. Existing runs, exports and projects keep working."
      />
      <Tabs value={kind} onChange={(v) => v && setKind(v)}>
        <Tabs.List>
          {KINDS.map((k) => (
            <Tabs.Tab key={k.value} value={k.value}>
              {k.label}
            </Tabs.Tab>
          ))}
        </Tabs.List>
      </Tabs>
      {/* keyed by kind so each tab starts with a clean filter */}
      <PluginTable key={kind} kind={kind} hint={current?.hint} />
    </div>
  )
}

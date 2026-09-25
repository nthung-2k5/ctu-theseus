/** Small dialogs for the filesystem view's folder actions on a not-yet-created class. */

import { Button, Group, Select, Stack, Text, TextInput } from '@mantine/core'
import { modals } from '@mantine/modals'
import { useState } from 'react'

function RenameForm({ name, onSubmit }: { name: string; onSubmit: (name: string) => void }) {
  const [value, setValue] = useState(name)
  const submit = () => {
    if (!value.trim()) return
    onSubmit(value.trim())
    modals.closeAll()
  }
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <Stack gap="sm">
        <TextInput label="Class name" value={value} onChange={(e) => setValue(e.currentTarget.value)} data-autofocus />
        <Text size="xs" c="dimmed">
          If a class with this name already exists, the files join it instead.
        </Text>
        <Group justify="flex-end">
          <Button variant="default" onClick={() => modals.closeAll()}>
            Cancel
          </Button>
          <Button type="submit" disabled={!value.trim()}>
            Rename
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

export function openRenameFolder(name: string, onSubmit: (name: string) => void) {
  modals.open({ title: 'Rename new class', children: <RenameForm name={name} onSubmit={onSubmit} /> })
}

function MapForm({
  options,
  onSubmit,
}: {
  options: { value: string; label: string }[]
  onSubmit: (classKey: string) => void
}) {
  const [value, setValue] = useState<string | null>(null)
  return (
    <Stack gap="sm">
      <Select
        label="Existing class"
        placeholder="Pick a class"
        data={options}
        value={value}
        onChange={setValue}
        searchable
        data-autofocus
      />
      <Group justify="flex-end">
        <Button variant="default" onClick={() => modals.closeAll()}>
          Cancel
        </Button>
        <Button
          disabled={!value}
          onClick={() => {
            if (value) onSubmit(value)
            modals.closeAll()
          }}
        >
          Move files
        </Button>
      </Group>
    </Stack>
  )
}

export function openMapToClass(
  folderName: string,
  options: { value: string; label: string }[],
  onSubmit: (classKey: string) => void,
) {
  modals.open({
    title: `Use an existing class for “${folderName}”`,
    children: <MapForm options={options} onSubmit={onSubmit} />,
  })
}

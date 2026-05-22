import { Button, ColorInput, Group, Stack, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { api } from '@public/lib/api'
import { assert } from '@public/lib/assert'
import { useEdenMutation } from '@public/lib/eden-query'
import type { DatasetClass } from '@public/store/types'
import { useQueryClient } from '@tanstack/react-query'

interface EditClassModalProps {
  projectId: string
  cls: DatasetClass
}

export const EditClassModal = ({ projectId, cls }: EditClassModalProps) => {
  const queryClient = useQueryClient()

  const form = useForm({
    initialValues: {
      name: cls.name,
      color: cls.color,
    },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Class name is required'),
      color: (v) => (/^#([0-9A-F]{3}){1,2}$/i.test(v) ? null : 'Enter a valid hex color'),
    },
  })

  const updateClass = useEdenMutation(api.classes({ classId: cls.id }).patch, {
    onSuccess: async ({ class: updated }) => {
      await queryClient.invalidateQueries({ queryKey: ['projects', projectId] })
      notifications.show({
        title: 'Class updated',
        message: `"${updated.name}" has been updated`,
        color: 'green',
      })
      modals.closeAll()
    },
    onError: (error) => {
      assert(error.status === 404 || error.status === 422)
      notifications.show({
        title: 'Error',
        message:
          error.status === 404
            ? typeof error.value === 'string'
              ? error.value
              : 'Class not found'
            : (error.value?.message ?? 'Failed to update class'),
        color: 'red',
      })
    },
    onSettled: () => form.reset(),
  })

  return (
    <form onSubmit={form.onSubmit((values) => updateClass.mutate(values))}>
      <Stack gap="md">
        <TextInput label="Class name" placeholder="e.g. Cat, Dog, Car" {...form.getInputProps('name')} />
        <ColorInput
          label="Color"
          format="hex"
          swatches={['#e03131', '#2f9e44', '#1971c2', '#f08c00', '#9c36b5', '#0c8599', '#e8590c', '#6741d9']}
          {...form.getInputProps('color')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={modals.closeAll} disabled={updateClass.isPending}>
            Cancel
          </Button>
          <Button type="submit" loading={updateClass.isPending}>
            Save changes
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

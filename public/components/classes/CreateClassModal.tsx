import { Button, ColorInput, Group, Stack, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { api } from '@public/lib/api'
import { useEdenMutation } from '@public/lib/eden-query'
import { useQueryClient } from '@tanstack/react-query'

const PRESET_COLORS = ['#e03131', '#2f9e44', '#1971c2', '#f08c00', '#9c36b5', '#0c8599', '#e8590c', '#6741d9']

interface CreateClassModalProps {
  projectId: string
  existingCount: number
}

export const CreateClassModal = ({ projectId, existingCount }: CreateClassModalProps) => {
  const queryClient = useQueryClient()

  const form = useForm({
    initialValues: {
      name: '',
      color: PRESET_COLORS[existingCount % PRESET_COLORS.length],
    },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Class name is required'),
      color: (v) => (/^#([0-9A-F]{3}){1,2}$/i.test(v) ? null : 'Enter a valid hex color'),
    },
  })

  const createClass = useEdenMutation(api.projects({ projectId }).classes.post, {
    onSuccess: async ({ class: cls }) => {
      await queryClient.invalidateQueries({ queryKey: ['projects', projectId] })
      notifications.show({
        title: 'Class created',
        message: `"${cls.name}" has been added`,
        color: 'green',
      })
      modals.closeAll()
    },
    onError: (error) => {
      notifications.show({
        title: 'Error',
        message: typeof error.value === 'string' ? error.value : (error.value?.message ?? 'Failed to create class'),
        color: 'red',
      })
    },
    onSettled: () => form.reset(),
  })

  return (
    <form onSubmit={form.onSubmit((values) => createClass.mutate(values))}>
      <Stack gap="md">
        <TextInput label="Class name" placeholder="e.g. Cat, Dog, Car" {...form.getInputProps('name')} />
        <ColorInput label="Color" format="hex" swatches={PRESET_COLORS} {...form.getInputProps('color')} />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={modals.closeAll} disabled={createClass.isPending}>
            Cancel
          </Button>
          <Button type="submit" loading={createClass.isPending}>
            Add class
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

import { Button, Group, Stack, Textarea, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { useEden } from '@public/lib/api'
import { assert } from '@public/lib/assert'
import { useMutation, useQueryClient } from '@tanstack/react-query'

/** Shared by DashboardPage's and ProjectPage's "Edit project" modals — same form, same mutation, same error handling. */
export function UpdateProjectModal({
  projectId,
  projectName,
  projectDescription,
  onDone,
}: {
  projectId: string
  projectName: string
  projectDescription: string | null
  onDone?: () => void
}) {
  const form = useForm({
    initialValues: { name: projectName, description: projectDescription },
    validate: {
      name: (v) => (v.trim().length > 0 ? null : 'Project name is required'),
    },
  })

  const eden = useEden()
  const queryClient = useQueryClient()

  const updateProject = useMutation({
    ...eden.api.projects({ projectId }).patch.mutationOptions(),
    onSuccess: ({ project }) => {
      queryClient.invalidateQueries({ queryKey: eden.api.projects.get.queryKey() })
      queryClient.invalidateQueries({ queryKey: eden.api.projects({ projectId }).get.queryKey() })
      notifications.show({ title: 'Project updated', message: `"${project.name}" has been updated`, color: 'green' })
      form.reset()
      onDone?.()
      modals.closeAll()
    },
    onError: (error) => {
      assert(error.status === 404 || error.status === 422)
      const value: unknown = error.value
      const message = error.status === 404 ? value : (value as { message?: string } | undefined)?.message
      notifications.show({
        title: 'Error',
        message: typeof message === 'string' ? message : 'Failed to update project',
        color: 'red',
      })
    },
  })

  return (
    <form onSubmit={form.onSubmit((values) => updateProject.mutate(values))}>
      <Stack gap="md">
        <TextInput label="Project name" placeholder="e.g. Traffic Signs" {...form.getInputProps('name')} />
        <Textarea
          label="Description"
          placeholder="What is this project about?"
          autosize
          minRows={3}
          {...form.getInputProps('description')}
        />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={modals.closeAll}>
            Cancel
          </Button>
          <Button type="submit" loading={updateProject.isPending}>
            Save Changes
          </Button>
        </Group>
      </Stack>
    </form>
  )
}

import { Button, Group, Stack, Textarea, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { apiErrorMessage } from '@public/lib/api/client'
import { getUpdateProjectMutationOptions } from '@public/lib/api/generated/projects/projects'
import { invalidateProjectList, invalidateProjectScope } from '@public/lib/queries'
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

  const queryClient = useQueryClient()

  const updateProject = useMutation({
    ...getUpdateProjectMutationOptions(),
    onSuccess: ({ project }) => {
      invalidateProjectList(queryClient)
      invalidateProjectScope(queryClient, projectId)
      notifications.show({ title: 'Project updated', message: `"${project.name}" has been updated`, color: 'green' })
      form.reset()
      onDone?.()
      modals.closeAll()
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to update project'), color: 'red' })
    },
  })

  return (
    <form onSubmit={form.onSubmit((values) => updateProject.mutate({ projectId, data: values }))}>
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

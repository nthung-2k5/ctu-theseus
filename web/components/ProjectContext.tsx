import { Center, Loader, Stack, Text } from '@mantine/core'
import { useProjectDetail } from '@public/queries/project'
import { useProjectStore } from '@public/store/useProjectStore'
import { type PropsWithChildren, useEffect } from 'react'
import { useParams } from 'wouter'

export const ProjectProvider = ({ children }: PropsWithChildren) => {
  const params = useParams<{ id: string }>()
  const setActiveProject = useProjectStore((s) => s.setActiveProject)

  const { data: project, isLoading, isSuccess } = useProjectDetail(params.id)

  useEffect(() => {
    return () => setActiveProject(null)
  }, [setActiveProject])

  useEffect(() => {
    if (isSuccess && project) {
      setActiveProject(project.project)
    }
  }, [project, isSuccess, setActiveProject])

  return isLoading ? (
    <Center>
      <Stack align="center" gap="sm">
        <Loader size="lg" color="primary" />
        <Text size="sm" c="dimmed">
          Loading project...
        </Text>
      </Stack>
    </Center>
  ) : (
    children
  )
}

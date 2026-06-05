import {
  Badge,
  Box,
  Button,
  Card,
  ColorSwatch,
  Group,
  Menu,
  ScrollArea,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { Dropzone, MIME_TYPES } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { CaretDownIcon, CloudArrowUpIcon } from '@phosphor-icons/react'
import { type QueuedFile, UploadItem } from '@public/components/upload_dataset/UploadItem'
import { api } from '@public/lib/api'
import { useProjectClasses } from '@public/queries/project'
import { useProjectStore } from '@public/store/useProjectStore'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useState } from 'react'
import { useParams } from 'wouter'

const MAX_FILE_SIZE = 5 * 1024 * 1024 // 5 MB

export const UploadDatasetPage = () => {
  const params = useParams<{ id: string }>()
  const projectId = params.id
  const activeProject = useProjectStore((s) => s.activeProject)
  const queryClient = useQueryClient()

  const [queue, setQueue] = useState<QueuedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  /* ── Fetch classes ── */
  const { data: classData, isLoading: classesLoading } = useProjectClasses(projectId)
  const classes = classData?.classes ?? []

  /* ── Add files to queue ── */
  const handleDrop = useCallback((files: File[]) => {
    const validFiles: QueuedFile[] = []
    const rejected: string[] = []

    for (const file of files) {
      if (file.size > MAX_FILE_SIZE) {
        rejected.push(`${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB > 5 MB limit)`)
        continue
      }
      validFiles.push({
        id: crypto.randomUUID(),
        file,
        preview: URL.createObjectURL(file),
        status: 'pending',
        progress: 0,
        classId: null,
      })
    }

    if (rejected.length > 0) {
      notifications.show({
        title: 'Some files were rejected',
        message: rejected.join('\n'),
        color: 'yellow',
        autoClose: 5000,
      })
    }

    setQueue((prev) => [...prev, ...validFiles])
  }, [])

  /* ── Class change for individual file ── */
  const handleClassChange = (id: string, classId: string | null) => {
    setQueue((prev) => prev.map((f) => (f.id === id ? { ...f, classId } : f)))
  }

  /* ── Remove a file from the queue ── */
  const removeFromQueue = useCallback((id: string) => {
    setQueue((prev) => {
      const item = prev.find((i) => i.id === id)
      if (item) URL.revokeObjectURL(item.preview)
      return prev.filter((i) => i.id !== id)
    })
  }, [])

  /* ── Upload all pending files ── */
  const startUpload = async () => {
    const pending = queue.filter((f) => f.status === 'pending' || f.status === 'error')
    if (pending.length === 0) return

    setIsUploading(true)

    // Reset errored items to pending
    setQueue((prev) =>
      prev.map((f) => (f.status === 'error' ? { ...f, status: 'pending' as const, progress: 0, error: undefined } : f)),
    )

    // Upload concurrently (max 10 at a time)
    const concurrency = 10
    let idx = 0

    const uploadOne = async (item: QueuedFile) => {
      // Mark uploading
      setQueue((prev) => prev.map((f) => (f.id === item.id ? { ...f, status: 'uploading' as const, progress: 10 } : f)))

      try {
        setQueue((prev) => prev.map((f) => (f.id === item.id ? { ...f, progress: 30 } : f)))

        const { data, error } = await api.projects({ projectId }).images.post({
          image: item.file,
          classId: item.classId ?? null,
        })

        if (error) {
          const msg = typeof error.value === 'string' ? error.value : ((error.value as any)?.message ?? 'Upload failed')
          throw new Error(msg)
        }

        setQueue((prev) => prev.map((f) => (f.id === item.id ? { ...f, progress: 90 } : f)))

        // The API returns images as PromiseSettledResult[]
        // Each entry: { status: 'fulfilled', value: DbRow } | { status: 'rejected', reason: any }
        const firstResult = data.images[0] as PromiseSettledResult<{ filename: string }> | undefined
        if (!firstResult) throw new Error('No response from server')
        if (firstResult.status === 'rejected') throw new Error((firstResult.reason as any)?.message ?? 'Server error')

        const serverRow = firstResult.value
        // Rename detection: server gave a different filename than the original
        const wasRenamed = serverRow.filename !== item.file.name

        setQueue((prev) =>
          prev.map((f) =>
            f.id === item.id
              ? {
                  ...f,
                  status: 'done' as const,
                  progress: 100,
                  renamed: wasRenamed,
                  serverName: serverRow.filename,
                }
              : f,
          ),
        )
      } catch (e: any) {
        setQueue((prev) =>
          prev.map((f) => (f.id === item.id ? { ...f, status: 'error' as const, progress: 0, error: e.message } : f)),
        )
      }
    }

    // Process in batches
    const items = [...pending]
    const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
      while (idx < items.length) {
        const current = items[idx++]
        await uploadOne(current)
      }
    })

    await Promise.all(workers)

    // Invalidate the images query so the dataset page is fresh
    await queryClient.invalidateQueries({ queryKey: ['images', projectId] })
    await queryClient.invalidateQueries({ queryKey: ['images-stats', projectId] })

    setIsUploading(false)

    const done = queue.filter((f) => f.status === 'done' || f.status === 'uploading').length
    const renamed = queue.filter((f) => f.renamed).length

    notifications.show({
      title: 'Upload complete',
      message: `${pending.length} file(s) processed${renamed > 0 ? ` (${renamed} renamed)` : ''}`,
      color: 'green',
    })
  }

  /* ── Clear completed from queue ── */
  const clearCompleted = () => {
    setQueue((prev) => {
      for (const item of prev) {
        if (item.status === 'done') URL.revokeObjectURL(item.preview)
      }
      return prev.filter((f) => f.status !== 'done')
    })
  }

  /* ── Clear all from queue ── */
  const clearAllQueue = () => {
    setQueue((prev) => {
      for (const item of prev) {
        URL.revokeObjectURL(item.preview)
      }
      return []
    })
  }

  /* ── Stats ── */
  const pendingCount = queue.filter((f) => f.status === 'pending').length
  const uploadingCount = queue.filter((f) => f.status === 'uploading').length
  const doneCount = queue.filter((f) => f.status === 'done').length
  const errorCount = queue.filter((f) => f.status === 'error').length
  const renamedCount = queue.filter((f) => f.renamed).length

  return (
    <Box>
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={2}>Upload Images</Title>
            <Text size="sm" c="dimmed" mt={4}>
              {activeProject ? `Add images to "${activeProject.name}"` : 'Upload images to the dataset'}
            </Text>
          </div>

          <Group gap="sm" align="flex-end">
            {classesLoading ? (
              <Skeleton height={36} width={200} radius="sm" />
            ) : (
              <Menu shadow="md" width={220} position="bottom-end">
                <Menu.Target>
                  <Button variant="light" color="gray" rightSection={<CaretDownIcon size={14} />}>
                    Assign class to all
                  </Button>
                </Menu.Target>
                <Menu.Dropdown>
                  <Menu.Item
                    onClick={() => setQueue((prev) => prev.map((f) => ({ ...f, classId: null })))}
                    leftSection={
                      <Box
                        w={14}
                        h={14}
                        style={{ borderRadius: '50%', border: '1px dashed var(--mantine-color-dimmed)' }}
                      />
                    }
                  >
                    <Text size="sm" c="dimmed">
                      Unclassified
                    </Text>
                  </Menu.Item>
                  <Menu.Divider />
                  {classes.map((c) => (
                    <Menu.Item
                      key={c.id}
                      onClick={() => setQueue((prev) => prev.map((f) => ({ ...f, classId: c.id })))}
                      leftSection={<ColorSwatch color={c.color} size={14} />}
                    >
                      {c.name}
                    </Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>
            )}
            <Button
              leftSection={<CloudArrowUpIcon size={18} />}
              onClick={startUpload}
              loading={isUploading}
              disabled={pendingCount === 0 && errorCount === 0}
            >
              Upload {pendingCount + errorCount > 0 ? `(${pendingCount + errorCount})` : ''}
            </Button>
            {doneCount > 0 && (
              <Button variant="light" color="green" onClick={clearCompleted}>
                Clear completed ({doneCount})
              </Button>
            )}
            {queue.length > 0 && (
              <Button variant="light" color="red" onClick={clearAllQueue} disabled={isUploading}>
                Clear all
              </Button>
            )}
          </Group>
        </Group>

        {/* Status summary */}
        {queue.length > 0 && (
          <Card withBorder padding="sm" radius="md">
            <Group gap="md" align="center" justify="flex-end">
              <Badge variant="light" color="gray" size="sm">
                {queue.length} total
              </Badge>
              {pendingCount > 0 && (
                <Badge variant="light" color="blue" size="sm">
                  {pendingCount} pending
                </Badge>
              )}
              {doneCount > 0 && (
                <Badge variant="light" color="green" size="sm">
                  {doneCount} done
                </Badge>
              )}
              {errorCount > 0 && (
                <Badge variant="light" color="red" size="sm">
                  {errorCount} failed
                </Badge>
              )}
              {renamedCount > 0 && (
                <Badge variant="light" color="yellow" size="sm">
                  {renamedCount} renamed
                </Badge>
              )}
            </Group>
          </Card>
        )}

        {/* Dropzone */}
        <Dropzone
          onDrop={handleDrop}
          accept={[MIME_TYPES.png, MIME_TYPES.jpeg]}
          maxSize={MAX_FILE_SIZE}
          radius="md"
          p={queue.length > 0 ? 'xs' : 'xl'}
          disabled={isUploading}
          activateOnClick={queue.length === 0}
          onReject={(rejections) => {
            const msgs = rejections.map((r) => {
              const err = r.errors[0]
              if (err?.code === 'file-too-large') return `${r.file.name}: exceeds 5 MB`
              if (err?.code === 'file-invalid-type') return `${r.file.name}: not an image`
              return `${r.file.name}: ${err?.message ?? 'rejected'}`
            })
            notifications.show({
              title: 'Files rejected',
              message: msgs.join('\n'),
              color: 'red',
            })
          }}
        >
          {queue.length === 0 ? (
            <Stack align="center" gap="sm" py="xl">
              <ThemeIcon size={48} variant="light" color="primary" radius="xl">
                <CloudArrowUpIcon size={28} />
              </ThemeIcon>
              <Text size="sm" fw={500}>
                Drag images here or click to browse
              </Text>
              <Text size="xs" c="dimmed">
                JPG, PNG — max 5 MB each
              </Text>
            </Stack>
          ) : (
            <ScrollArea.Autosize mah="calc(100vh - 280px)" type="auto">
              <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 6 }} spacing="sm" style={{ pointerEvents: 'all' }}>
                {queue.map((item) => (
                  <UploadItem
                    key={item.id}
                    item={item}
                    classes={classes}
                    onRemove={removeFromQueue}
                    onClassChange={handleClassChange}
                  />
                ))}
              </SimpleGrid>
            </ScrollArea.Autosize>
          )}
        </Dropzone>
      </Stack>
    </Box>
  )
}

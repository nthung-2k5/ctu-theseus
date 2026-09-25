/**
 * Summary + Upload/Discard controls for the filesystem view.
 *
 * Says what is about to happen — counts per split, what won't be sent, which classes will be created — and
 * runs the upload, chunk by chunk, keeping refused files staged with the reason.
 */

import { Badge, Button, Group, Progress, Stack, Text, ThemeIcon, Title, Tooltip } from '@mantine/core'
import { modals } from '@mantine/modals'
import { notifications } from '@mantine/notifications'
import { ListChecksIcon, TrashIcon, UploadSimpleIcon } from '@phosphor-icons/react'
import { SPLIT_OPTIONS } from '@public/lib/constants'
import { invalidateProjectScope } from '@public/lib/queries'
import { isUploadable } from '@public/lib/upload/fileTree'
import { type UploadProgress, uploadStaged } from '@public/lib/upload/uploadStaged'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'

function plural(n: number, word: string) {
  return `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`
}

export function FileUploadBar({
  projectId,
  staging,
  taskUsesClasses,
  onUploadingChange,
}: {
  projectId: string
  staging: FileStaging
  taskUsesClasses: boolean
  onUploadingChange: (uploading: boolean) => void
}) {
  const queryClient = useQueryClient()
  const { state, index, busy, remove, clear, clearServerErrors, serverErrors } = staging
  const { totals } = index

  const [progress, setProgress] = useState<UploadProgress | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const upload = useMutation({
    mutationFn: async () => {
      const files = [...state.files.values()].filter(isUploadable)
      const controller = new AbortController()
      abortRef.current = controller
      onUploadingChange(true)
      clearServerErrors()
      try {
        return await uploadStaged({
          projectId,
          files,
          pending: state.pending,
          signal: controller.signal,
          onProgress: setProgress,
          onSettled: (uploadedIds, errors) => {
            remove(uploadedIds)
            serverErrors(errors)
          },
        })
      } finally {
        abortRef.current = null
        setProgress(null)
        onUploadingChange(false)
        // New classes may have been created; let the class list (and pending-folder reconciliation) catch up.
        void invalidateProjectScope(queryClient, projectId)
      }
    },
    onSuccess: (outcome) => {
      const details = [
        outcome.createdClasses > 0 && `${plural(outcome.createdClasses, 'class')} created`,
        outcome.duplicates > 0 && `${outcome.duplicates.toLocaleString()} already in the pool`,
      ].filter(Boolean)
      const suffix = details.length > 0 ? ` (${details.join(', ')})` : ''
      if (outcome.cancelled) {
        notifications.show({
          title: 'Upload cancelled',
          message: `${plural(outcome.uploaded, 'file')} added before it stopped${suffix}`,
          color: 'yellow',
        })
      } else if (outcome.failed > 0) {
        notifications.show({
          title: 'Upload partially complete',
          message: `${plural(outcome.uploaded, 'file')} added, ${outcome.failed.toLocaleString()} failed and stayed in the tree${suffix}`,
          color: 'yellow',
        })
      } else {
        notifications.show({
          title: 'Upload complete',
          message: `${plural(outcome.uploaded, 'file')} added to the pool${suffix}`,
          color: 'green',
        })
      }
    },
    onError: () => {
      notifications.show({ title: 'Upload failed', message: 'Nothing more was sent', color: 'red' })
    },
  })

  const uploading = upload.isPending
  const newClassNames = totals.newClassNames

  const startUpload = () => {
    if (newClassNames.length === 0) {
      upload.mutate()
      return
    }
    modals.openConfirmModal({
      title: `Create ${plural(newClassNames.length, 'new class')}?`,
      children: (
        <Stack gap="xs">
          <Text size="sm">
            These folders don't match an existing class. Uploading creates a class for each one and labels its files
            with it.
          </Text>
          <Group gap={6}>
            {newClassNames.slice(0, 30).map((name) => (
              <Badge key={name} variant="light" color="teal">
                {name}
              </Badge>
            ))}
            {newClassNames.length > 30 && <Text size="xs">…and {newClassNames.length - 30} more</Text>}
          </Group>
          <Text size="xs" c="dimmed">
            To use an existing class instead, cancel and choose "Use an existing class…" from the folder's menu.
          </Text>
        </Stack>
      ),
      labels: { confirm: 'Create and upload', cancel: 'Cancel' },
      onConfirm: () => upload.mutate(),
    })
  }

  return (
    <Stack gap="xs">
      <Group justify="space-between" wrap="nowrap">
        <Group gap="sm">
          <ThemeIcon size="md" variant="light" color="primary">
            <ListChecksIcon size={18} />
          </ThemeIcon>
          <Title order={5}>Ready to upload</Title>
          {totals.files > 0 && (
            <Badge size="sm" variant="light">
              {totals.files.toLocaleString()}
            </Badge>
          )}
        </Group>
        <Group gap="xs" wrap="nowrap">
          {uploading ? (
            <Button size="xs" variant="subtle" color="red" onClick={() => abortRef.current?.abort()}>
              Cancel upload
            </Button>
          ) : (
            totals.files > 0 && (
              <Tooltip label="Discard everything staged — nothing has been uploaded yet">
                <Button
                  size="xs"
                  variant="subtle"
                  color="red"
                  leftSection={<TrashIcon size={14} />}
                  onClick={clear}
                  disabled={busy !== null}
                >
                  Discard
                </Button>
              </Tooltip>
            )
          )}
          <Button
            leftSection={<UploadSimpleIcon size={14} />}
            onClick={startUpload}
            loading={uploading}
            disabled={totals.uploadable === 0 || busy !== null}
          >
            {totals.uploadable > 0 ? `Upload ${plural(totals.uploadable, 'file')}` : 'Upload'}
          </Button>
        </Group>
      </Group>

      {uploading && progress && (
        <Group gap="sm" wrap="nowrap">
          <Progress value={(progress.settled / Math.max(1, progress.total)) * 100} style={{ flex: 1 }} animated />
          <Text size="xs" c="dimmed">
            {progress.settled.toLocaleString()} / {progress.total.toLocaleString()}
          </Text>
        </Group>
      )}

      {totals.files > 0 && (
        <Group gap="md" wrap="wrap">
          {SPLIT_OPTIONS.map(({ value, label }) => (
            <Text key={value} size="xs" c="dimmed">
              {label}:{' '}
              <Text component="span" fw={600} c="var(--mantine-color-text)">
                {totals.bySplit[value as keyof typeof totals.bySplit].toLocaleString()}
              </Text>
            </Text>
          ))}
          {totals.withErrors > 0 && (
            <Text size="xs" c="red.7">
              {plural(totals.withErrors, 'file')} with errors won't be uploaded
            </Text>
          )}
          {totals.serverFailures > 0 && (
            <Text size="xs" c="red.7">
              {plural(totals.serverFailures, 'file')} refused by the server — hover a row for the reason, or press
              Upload to retry
            </Text>
          )}
          {totals.withWarnings > 0 && (
            <Text size="xs" c="yellow.7">
              {plural(totals.withWarnings, 'file')} with warnings
            </Text>
          )}
          {taskUsesClasses && totals.unclassified > 0 && (
            <Text size="xs" c="yellow.7">
              {totals.unclassified.toLocaleString()} without a class
            </Text>
          )}
          {newClassNames.length > 0 && (
            <Text size="xs" c="teal.7">
              {plural(newClassNames.length, 'new class')} will be created
            </Text>
          )}
        </Group>
      )}
    </Stack>
  )
}

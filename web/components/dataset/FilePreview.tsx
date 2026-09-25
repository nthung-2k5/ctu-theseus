/** Side preview for the single selected file in the filesystem view: thumbnail or player, plus its problems. */

import { Badge, Group, Stack, Text } from '@mantine/core'
import { formatBytes } from '@public/lib/format'
import { fileIssues } from '@public/lib/upload/fileTree'
import type { StagedFile } from '@public/lib/upload/types'
import { useEffect, useState } from 'react'

export function FilePreview({ file, taskUsesClasses }: { file: StagedFile; taskUsesClasses: boolean }) {
  const kind = file.file.type.startsWith('image/') ? 'image' : file.file.type.startsWith('audio/') ? 'audio' : null

  // The object URL exists only while this file is selected, so browsing a large tree never holds blobs open.
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!kind) return
    const objectUrl = URL.createObjectURL(file.file)
    setUrl(objectUrl)
    return () => {
      URL.revokeObjectURL(objectUrl)
      setUrl(null)
    }
  }, [file.file, kind])

  const issues = fileIssues(file, taskUsesClasses)
  return (
    <Stack gap="xs">
      {kind === 'image' && url && (
        <img
          src={url}
          alt={file.name}
          style={{
            width: '100%',
            maxHeight: 220,
            objectFit: 'contain',
            borderRadius: 6,
            background: 'var(--mantine-color-default-hover)',
          }}
        />
      )}
      {kind === 'audio' && url && (
        // biome-ignore lint/a11y/useMediaCaption: dataset audio has no captions to offer
        <audio src={url} controls style={{ width: '100%' }} />
      )}
      <div>
        <Text size="sm" fw={600} style={{ wordBreak: 'break-all' }}>
          {file.name}
        </Text>
        <Text size="xs" c="dimmed">
          {formatBytes(file.size)}
          {file.sourceDirs.length > 0 && ` · from ${file.sourceDirs.join('/')}`}
        </Text>
      </div>
      {issues.length > 0 && (
        <Stack gap={4}>
          {issues.map((issue) => (
            <Group key={issue.message} gap={6} wrap="nowrap" align="flex-start">
              <Badge size="xs" variant="light" color={issue.severity === 'error' ? 'red' : 'yellow'}>
                {issue.severity}
              </Badge>
              <Text size="xs">{issue.message}</Text>
            </Group>
          ))}
        </Stack>
      )}
    </Stack>
  )
}

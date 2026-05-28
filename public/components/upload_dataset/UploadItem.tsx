import { ActionIcon, Box, Card, ColorSwatch, Group, Image, Progress, Select, Stack, Text, Tooltip } from '@mantine/core'
import { CheckCircleIcon, SpinnerIcon, TrashIcon, WarningIcon, XCircleIcon } from '@phosphor-icons/react'

export type UploadStatus = 'pending' | 'uploading' | 'done' | 'error'

export interface QueuedFile {
  id: string
  file: File
  preview: string
  status: UploadStatus
  progress: number
  error?: string
  renamed?: boolean
  serverName?: string
  classId?: string | null
}

interface UploadItemProps {
  item: QueuedFile
  classes: { id: string; name: string; color: string }[]
  onRemove: (id: string) => void
  onClassChange: (id: string, classId: string | null) => void
}

export const UploadItem = ({ item, classes, onRemove, onClassChange }: UploadItemProps) => {
  return (
    <Card
      radius="sm"
      p={0}
      withBorder
      style={{
        opacity: item.status === 'error' ? 0.8 : 1,
        overflow: 'visible',
      }}
    >
      {/* Image Area */}
      <Box pos="relative" style={{ aspectRatio: '4/3', overflow: 'hidden' }}>
        <Image src={item.preview} alt={item.file.name} fit="cover" w="100%" h="100%" />

        {/* Status Overlay */}
        <Box pos="absolute" top={4} left={4} style={{ zIndex: 2 }}>
          <Group gap={4}>
            {item.status === 'done' && <CheckCircleIcon size={20} weight="fill" color="var(--mantine-color-green-5)" />}
            {item.status === 'error' && (
              <Tooltip label={item.error} withArrow maw={300} multiline>
                <XCircleIcon size={20} weight="fill" color="var(--mantine-color-red-5)" />
              </Tooltip>
            )}
            {item.status === 'uploading' && <SpinnerIcon size={20} className="spin" color="var(--mantine-primary-color-5)" />}
            {item.renamed && (
              <Tooltip label={`Renamed to "${item.serverName}"`} withArrow>
                <WarningIcon size={20} weight="fill" color="var(--mantine-color-yellow-5)" />
              </Tooltip>
            )}
          </Group>
        </Box>

        {/* Delete Button (Top Right) */}
        {(item.status === 'pending' || item.status === 'error') && (
          <ActionIcon
            pos="absolute"
            top={4}
            right={4}
            size="sm"
            variant="filled"
            color="red"
            style={{ zIndex: 2, boxShadow: '0 2px 4px rgba(0,0,0,0.2)' }}
            onClick={(e) => {
              e.stopPropagation()
              onRemove(item.id)
            }}
          >
            <TrashIcon size={14} />
          </ActionIcon>
        )}
      </Box>

      {/* Progress Bar */}
      {item.status === 'uploading' && (
        <Progress value={item.progress} size="xs" color="primary" animated radius={0} />
      )}

      {/* Info & Actions */}
      <Stack p="xs" gap="xs">
        <Stack gap={2}>
          <Text size="xs" fw={500} truncate="end" title={item.renamed ? item.serverName : item.file.name}>
            {item.renamed ? item.serverName : item.file.name}
          </Text>
          <Text size="xs" c="dimmed">
            {(item.file.size / 1024).toFixed(0)} KB
          </Text>
        </Stack>

        <Select
          placeholder="Assign class"
          clearable
          size="xs"
          data={classes.map((c) => ({
            value: c.id,
            label: c.name,
          }))}
          value={item.classId || null}
          onChange={(v) => {
            onClassChange(item.id, v)
          }}
          disabled={item.status === 'uploading' || item.status === 'done'}
          onClick={(e) => e.stopPropagation()}
          leftSection={
            item.classId ? (
              <ColorSwatch color={classes.find((c) => c.id === item.classId)?.color ?? '#888'} size={10} />
            ) : undefined
          }
        />
      </Stack>
    </Card>
  )
}

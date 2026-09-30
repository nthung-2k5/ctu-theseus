import { Group, Paper, Stack, Text, ThemeIcon } from '@mantine/core'
import { Dropzone, type FileRejection } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { CloudArrowUpIcon, XIcon } from '@phosphor-icons/react'
import { formatBytes } from '@public/lib/customModels'

const ACCEPTED = /\.(zip|safetensors)$/i

const validate = (file: File) =>
  ACCEPTED.test(file.name)
    ? null
    : { code: 'bad-extension', message: 'Use a .zip of the model folder or a .safetensors file' }

/**
 * Pick the weights: a .zip of a Hugging Face model folder, or a single .safetensors file. Only the extension
 * is checked here; the server checks the rest (size, contents, no pickles) when it validates the upload.
 */
export function ModelFileDropzone({
  file,
  onChange,
  disabled,
}: {
  file: File | null
  onChange: (file: File | null) => void
  disabled?: boolean
}) {
  const onReject = (rejections: FileRejection[]) =>
    notifications.show({
      title: 'Not a model file',
      message: rejections[0]?.errors[0]?.message ?? 'Use a .zip or .safetensors file',
      color: 'red',
    })

  if (file) {
    return (
      <Paper withBorder p="sm" radius="md">
        <Group justify="space-between" wrap="nowrap">
          <div style={{ minWidth: 0 }}>
            <Text size="sm" truncate>
              {file.name}
            </Text>
            <Text size="xs" c="dimmed">
              {formatBytes(file.size)}
            </Text>
          </div>
          {!disabled && (
            <XIcon size={16} style={{ cursor: 'pointer' }} aria-label="Remove file" onClick={() => onChange(null)} />
          )}
        </Group>
      </Paper>
    )
  }

  return (
    <Dropzone
      multiple={false}
      validator={validate}
      onDrop={(files) => onChange(files[0] ?? null)}
      onReject={onReject}
      radius="md"
      p="lg"
      style={{ borderStyle: 'dashed', borderWidth: 2 }}
    >
      <Stack align="center" gap={6}>
        <ThemeIcon size={44} variant="light" color="primary" radius="xl">
          <CloudArrowUpIcon size={24} />
        </ThemeIcon>
        <Text size="sm" fw={500}>
          Drop a model here or click to browse
        </Text>
        <Text size="xs" c="dimmed" ta="center">
          A .zip of the model folder (config.json, tokenizer and .safetensors weights) or a single .safetensors file.
          Large files upload directly to storage.
        </Text>
      </Stack>
    </Dropzone>
  )
}

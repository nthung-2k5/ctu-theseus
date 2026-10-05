import { Group, Loader, Text } from '@mantine/core'

/**
 * Shown while a snapshot builds. The server only reports a status (building | ready | failed), never a
 * percentage, so this says what is happening instead of inventing a progress bar; the project query polls
 * while any snapshot is building, so the page updates by itself when it finishes.
 */
export const BuildStatus = () => (
  <Group gap={6} role="status">
    <Loader size={12} />
    <Text size="xs" c="dimmed">
      Building. This page updates when the snapshot is ready.
    </Text>
  </Group>
)

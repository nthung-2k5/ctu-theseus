import { ActionIcon, CopyButton, Group, Text, Tooltip } from '@mantine/core'
import { CheckIcon, CopyIcon } from '@phosphor-icons/react'
import { MONO_STACK } from '@public/theme'

/** A monospace value with a copy button (ids, hashes). */
export function CopyField({ value, truncate = true }: { value: string; truncate?: boolean }) {
  return (
    <Group gap={4} wrap="nowrap">
      <Text size="xs" truncate={truncate ? 'end' : undefined} style={{ fontFamily: MONO_STACK }}>
        {value}
      </Text>
      <CopyButton value={value} timeout={1500}>
        {({ copied, copy }) => (
          <Tooltip label={copied ? 'Copied' : 'Copy'}>
            <ActionIcon size="xs" variant="subtle" color={copied ? 'teal' : 'gray'} onClick={copy} aria-label="Copy">
              {copied ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
    </Group>
  )
}

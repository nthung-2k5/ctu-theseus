import { Badge, Group, Tooltip, UnstyledButton } from '@mantine/core'
import type { LabelClass } from '@public/store/types'
import { useEffect } from 'react'

/**
 * The class-swatch picker shared by every modality's labeling surface.
 * Classes 1-9 get a keyboard shortcut (digit key) while `active` is true —
 * callers gate `active` on "an item is focused/selected" so digit presses
 * elsewhere on the page (e.g. typing in a search box) don't fire it.
 */
export function ClassPalette({
  classes,
  onSelect,
  active = true,
  size = 'sm',
}: {
  classes: LabelClass[]
  onSelect: (classId: string) => void
  active?: boolean
  size?: 'xs' | 'sm' | 'md'
}) {
  useEffect(() => {
    if (!active) return
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const digit = Number(e.key)
      if (Number.isInteger(digit) && digit >= 1 && digit <= 9 && classes[digit - 1]) {
        e.preventDefault()
        onSelect(classes[digit - 1].classId)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [active, classes, onSelect])

  const dims = { xs: 14, sm: 18, md: 22 }[size]

  return (
    <Group gap={6}>
      {classes.map((cls, i) => (
        <Tooltip key={cls.classId} label={i < 9 ? `${cls.name} (${i + 1})` : cls.name}>
          <UnstyledButton onClick={() => onSelect(cls.classId)}>
            <Badge
              size={size}
              variant="light"
              style={{
                backgroundColor: `${cls.uiColorHex ?? 'var(--mantine-color-gray-6)'}22`,
                color: cls.uiColorHex ?? undefined,
                border: `1px solid ${cls.uiColorHex ?? 'var(--mantine-color-gray-6)'}`,
                cursor: 'pointer',
              }}
              leftSection={
                <span
                  style={{
                    display: 'inline-block',
                    width: dims * 0.4,
                    height: dims * 0.4,
                    borderRadius: '50%',
                    backgroundColor: cls.uiColorHex ?? 'var(--mantine-color-gray-6)',
                  }}
                />
              }
            >
              {cls.name}
            </Badge>
          </UnstyledButton>
        </Tooltip>
      ))}
    </Group>
  )
}

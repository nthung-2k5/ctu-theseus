import { Text, type TextProps } from '@mantine/core'
import type { ReactNode } from 'react'

/** The uppercase dimmed xs label used above sections and inside KPI tiles. */
export function SectionLabel({ children, ...props }: { children: ReactNode } & Omit<TextProps, 'children'>) {
  return (
    <Text size="xs" c="dimmed" tt="uppercase" fw={500} {...props}>
      {children}
    </Text>
  )
}

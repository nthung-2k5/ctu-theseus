/**
 * A small "i" icon that explains a setting in plain words on hover, keyboard focus or touch. Used on the form
 * options of New experiment so a setting's name is never the only thing a non-technical user has to go on.
 */

import { Tooltip } from '@mantine/core'
import { InfoIcon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'

export function InfoTip({ text }: { text: ReactNode }) {
  // Focusable so keyboard users can reach it (the tooltip opens on focus). It usually sits inside a <label>, where
  // a plain click would toggle the switch or focus the input the label belongs to, so clicks are swallowed here.
  // A <button> would be the usual way to get both, but a labelable element is not valid inside a <label>.
  const icon = (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the click handler only stops the label's own activation
    <span
      // biome-ignore lint/a11y/noNoninteractiveTabindex: focusable on purpose, so the tooltip works from the keyboard
      tabIndex={0}
      role="img"
      aria-label={typeof text === 'string' ? text : 'More information'}
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
      }}
      style={{ display: 'inline-flex', color: 'var(--mantine-color-dimmed)', cursor: 'help' }}
    >
      <InfoIcon size={13} />
    </span>
  )

  return (
    <Tooltip
      label={text}
      multiline
      w={280}
      withArrow
      position="top-start"
      events={{ hover: true, focus: true, touch: true }}
    >
      {icon}
    </Tooltip>
  )
}

/** A field label followed by its InfoTip. With no `tip` it is just the label, so callers can pass an optional one. */
export function LabelWithTip({ label, tip }: { label: ReactNode; tip?: ReactNode }) {
  if (!tip) return <>{label}</>
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {label}
      <InfoTip text={tip} />
    </span>
  )
}

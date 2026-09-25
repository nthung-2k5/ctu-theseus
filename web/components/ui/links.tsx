import { Button, type ButtonProps, Card, type CardProps } from '@mantine/core'
import { createLink } from '@tanstack/react-router'
import { type AnchorHTMLAttributes, forwardRef } from 'react'

/**
 * Mantine components wrapped with TanStack's `createLink`, so `to`, `params` and `search` stay
 * type-checked against the route tree (a bare `component={Link}` loses that).
 */
type Anchor<P> = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, keyof P | 'color'>

const ButtonAnchor = forwardRef<HTMLAnchorElement, Omit<ButtonProps, 'component'> & Anchor<ButtonProps>>(
  (props, ref) => <Button component="a" ref={ref} {...props} />,
)
ButtonAnchor.displayName = 'ButtonAnchor'
export const LinkButton = createLink(ButtonAnchor)

const CardAnchor = forwardRef<HTMLAnchorElement, Omit<CardProps, 'component'> & Anchor<CardProps>>((props, ref) => (
  <Card component="a" ref={ref} {...props} />
))
CardAnchor.displayName = 'CardAnchor'
export const LinkCard = createLink(CardAnchor)

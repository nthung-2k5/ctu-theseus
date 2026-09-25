'use no memo'

import { ActionIcon, Button, Chip, Group, Text, TextInput, Tooltip } from '@mantine/core'
import { ArrowLineDownIcon, MagnifyingGlassIcon, PauseIcon } from '@phosphor-icons/react'
import type { LogLine } from '@public/hooks/useRunEvents'
import { parseAnsi } from '@public/lib/ansi'
import { MONO_STACK } from '@public/theme'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useEffect, useMemo, useRef, useState } from 'react'

const ROW_H = 20
const LEVELS: LogLine['level'][] = ['info', 'warn', 'error']
const LEVEL_COLOR: Record<LogLine['level'], string> = {
  info: '#00afef',
  warn: '#fbbf24',
  error: '#f87171',
}

/** Compile the search box: `/regex/` or plain (case-insensitive) text. Invalid regex falls back to text. */
function compileQuery(query: string): ((s: string) => boolean) | null {
  const q = query.trim()
  if (!q) return null
  const m = q.match(/^\/(.+)\/([a-z]*)$/)
  if (m) {
    try {
      const re = new RegExp(m[1], m[2] || 'i')
      return (s) => re.test(s)
    } catch {
      /* fall through to substring */
    }
  }
  const needle = q.toLowerCase()
  return (s) => s.toLowerCase().includes(needle)
}

/**
 * Virtualised, ANSI-aware log viewer: line-number gutter, level tags, level chips, search and
 * follow/pause. Lines come from `useRunEvents` (a capped ring) or, for finished runs, the uploaded log file.
 */
export function LogConsole({
  lines,
  height = '100%',
  empty = 'Waiting for output…',
  statusSlot,
}: {
  lines: LogLine[]
  height?: number | string
  empty?: string
  statusSlot?: React.ReactNode
}) {
  const [levels, setLevels] = useState<LogLine['level'][]>(LEVELS)
  const [query, setQuery] = useState('')
  const [follow, setFollow] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  const visible = useMemo(() => {
    const match = compileQuery(query)
    const out: { n: number; line: LogLine }[] = []
    lines.forEach((line, i) => {
      if (!levels.includes(line.level)) return
      if (match && !match(line.line)) return
      out.push({ n: i + 1, line })
    })
    return out
  }, [lines, levels, query])

  const virtualizer = useVirtualizer({
    count: visible.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_H,
    overscan: 12,
  })

  // Pin to the newest line while following.
  useEffect(() => {
    if (follow && visible.length > 0) virtualizer.scrollToIndex(visible.length - 1, { align: 'end' })
  }, [follow, visible.length, virtualizer])

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < ROW_H
    if (atBottom !== follow) setFollow(atBottom)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height, minHeight: 0 }}>
      <Group gap="xs" p={6} wrap="nowrap" style={{ borderBottom: '1px solid var(--mantine-color-default-border)' }}>
        <TextInput
          size="xs"
          leftSection={<MagnifyingGlassIcon size={13} />}
          placeholder="filter · text or /regex/"
          value={query}
          onChange={(e) => setQuery(e.currentTarget.value)}
          style={{ flex: '1 1 160px', minWidth: 120 }}
          styles={{ input: { fontFamily: MONO_STACK } }}
        />
        <Chip.Group multiple value={levels} onChange={(v) => setLevels(v as LogLine['level'][])}>
          <Group gap={4} wrap="nowrap">
            {LEVELS.map((l) => (
              <Chip key={l} value={l} size="xs" variant="light">
                {l}
              </Chip>
            ))}
          </Group>
        </Chip.Group>
        <Text size="xs" c="dimmed" className="tnum" style={{ whiteSpace: 'nowrap' }}>
          {lines.length.toLocaleString()} lines
        </Text>
        {statusSlot}
        <Tooltip label={follow ? 'Following: click to freeze' : 'Frozen: click to follow'}>
          <ActionIcon
            variant={follow ? 'filled' : 'light'}
            color={follow ? 'primary' : 'yellow'}
            onClick={() => setFollow((f) => !f)}
            aria-label="Toggle auto-scroll"
          >
            {follow ? <ArrowLineDownIcon size={15} /> : <PauseIcon size={15} />}
          </ActionIcon>
        </Tooltip>
      </Group>

      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        <div
          ref={scrollRef}
          onScroll={onScroll}
          style={{
            height: '100%',
            overflow: 'auto',
            background: 'var(--mantine-color-dark-9)',
            fontFamily: MONO_STACK,
            fontSize: 12,
            color: '#cbd5e1',
          }}
        >
          {visible.length === 0 ? (
            <Text size="xs" c="dimmed" p="sm" style={{ fontFamily: MONO_STACK }}>
              {lines.length === 0 ? empty : 'No lines match the filter.'}
            </Text>
          ) : (
            <div style={{ height: virtualizer.getTotalSize(), position: 'relative', minWidth: 'max-content' }}>
              {virtualizer.getVirtualItems().map((vi) => {
                const row = visible[vi.index]
                return (
                  <div
                    key={vi.key}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      height: ROW_H,
                      lineHeight: `${ROW_H}px`,
                      transform: `translateY(${vi.start}px)`,
                      whiteSpace: 'pre',
                      paddingRight: 12,
                      borderLeft: `2px solid ${row.line.level === 'error' ? '#f87171' : 'transparent'}`,
                    }}
                  >
                    <span style={{ color: '#7c8aa0', userSelect: 'none', marginRight: 10 }}>
                      {String(row.n).padStart(5, ' ')}
                    </span>
                    <span
                      style={{
                        color: LEVEL_COLOR[row.line.level],
                        marginRight: 8,
                        userSelect: 'none',
                      }}
                    >
                      {row.line.level.slice(0, 4).toUpperCase().padEnd(4, ' ')}
                    </span>
                    {parseAnsi(row.line.line).map((s, i) => (
                      <span
                        // biome-ignore lint/suspicious/noArrayIndexKey: spans are static per line
                        key={i}
                        style={{
                          color: s.color,
                          fontWeight: s.bold ? 600 : undefined,
                          opacity: s.dim ? 0.6 : undefined,
                        }}
                      >
                        {s.text}
                      </span>
                    ))}
                  </div>
                )
              })}
            </div>
          )}
        </div>
        {!follow && visible.length > 0 && (
          <Button
            size="compact-xs"
            color="cyan"
            style={{ position: 'absolute', right: 16, bottom: 12 }}
            leftSection={<ArrowLineDownIcon size={13} />}
            onClick={() => setFollow(true)}
          >
            Jump to latest
          </Button>
        )}
      </div>
    </div>
  )
}

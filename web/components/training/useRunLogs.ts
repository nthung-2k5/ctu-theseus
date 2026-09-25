import type { LogLine } from '@public/hooks/useRunEvents'
import { useQuery } from '@tanstack/react-query'
import { useRunContext } from './RunContext'

const levelOf = (line: string): LogLine['level'] => {
  if (/\b(error|exception|traceback|fatal)\b/i.test(line)) return 'error'
  if (/\bwarn(ing)?\b/i.test(line)) return 'warn'
  return 'info'
}

/**
 * The lines to show for a run: the live SSE ring while it is active, or the log file the worker
 * uploaded (GET /api/runs/:id/logs) once it has finished. A missing file just means no lines.
 */
export function useRunLogs(): { lines: LogLine[]; isLoading: boolean } {
  const { run, isActive, live } = useRunContext()

  const { data, isLoading } = useQuery({
    queryKey: ['run-logs', run.id],
    enabled: !isActive,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
    queryFn: async (): Promise<LogLine[]> => {
      const res = await fetch(`/api/runs/${run.id}/logs`, { credentials: 'include' })
      if (!res.ok) return []
      const text = await res.text()
      return text
        .split(/\r?\n/)
        .filter((l) => l.length > 0)
        .map((line) => ({ ts: '', level: levelOf(line), line }))
    },
  })

  return isActive ? { lines: live.logs, isLoading: false } : { lines: data ?? [], isLoading }
}

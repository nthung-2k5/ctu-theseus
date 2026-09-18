/**
 * ErrorAnalysisGrid – misclassified test-split items for a finished run,
 * most-confident-mistake first (a confident wrong answer is the most
 * informative one — see ai_service/services/evaluate.py's `_top_errors`).
 * Each row renders the item's own content (image/audio/text) alongside
 * actual vs. predicted label, so a suspected mislabel can be spotted without
 * leaving this tab — see server/routes/training.ts's
 * GET /runs/:runId/evaluation/errors.
 */

import { Badge, Button, Card, Group, Image, Pagination, Select, Stack, Text, Title } from '@mantine/core'
import { ArrowLeftIcon } from '@phosphor-icons/react'
import { EmptyState, QueryBoundary } from '@public/components/ui'
import { useLabelClasses, useRunEvaluationErrors } from '@public/lib/queries'
import type { EvaluationErrorRow, Modality, TrainingRunSummary } from '@public/store/types'
import { useState } from 'react'

function ErrorItemPreview({ row, modality }: { row: EvaluationErrorRow; modality: Modality | undefined }) {
  if (row.item?.text) {
    return (
      <Text size="sm" lineClamp={3} maw={280}>
        {row.item.text}
      </Text>
    )
  }
  if (row.item?.downloadUrl) {
    if (modality === 'vision') {
      return <Image src={row.item.downloadUrl} alt="" h={90} w={90} fit="cover" radius="sm" fallbackSrc="" />
    }
    if (modality === 'audio') {
      // biome-ignore lint/a11y/useMediaCaption: raw audio dataset item has no transcript source
      return <audio controls src={row.item.downloadUrl} style={{ maxWidth: 240 }} />
    }
  }
  return (
    <Text size="xs" c="dimmed">
      Item {row.itemId.slice(0, 8)}…
    </Text>
  )
}

function ErrorCard({ row, modality }: { row: EvaluationErrorRow; modality: Modality | undefined }) {
  return (
    <Card withBorder p="sm" radius="md">
      <Group justify="space-between" align="flex-start" wrap="nowrap">
        <ErrorItemPreview row={row} modality={modality} />
        <Stack gap={4} align="flex-end" style={{ flexShrink: 0 }}>
          <Group gap={6}>
            <Badge size="xs" variant="light" color="gray">
              actual
            </Badge>
            <Text size="sm" fw={600}>
              {row.actual}
            </Text>
          </Group>
          <Group gap={6}>
            <Badge size="xs" variant="light" color="red">
              predicted
            </Badge>
            <Text size="sm" fw={600} c="red">
              {row.predicted}
            </Text>
          </Group>
          {row.confidence !== null && (
            <Text size="xs" c="dimmed">
              {(row.confidence * 100).toFixed(1)}% confidence
            </Text>
          )}
        </Stack>
      </Group>
    </Card>
  )
}

export function ErrorAnalysisGrid({
  run,
  projectId,
  modality,
  classNames,
  onBack,
}: {
  run: TrainingRunSummary
  projectId: string
  modality: Modality | undefined
  /** Class names present in this run's confusion matrix (report.idx2str) — filter options, not every project class. */
  classNames: string[]
  onBack: () => void
}) {
  const [page, setPage] = useState(1)
  const [classId, setClassId] = useState<string | undefined>(undefined)

  const { data: classesData } = useLabelClasses(projectId)
  const classOptions = (classesData?.classes ?? [])
    .filter((c) => classNames.includes(c.name))
    .map((c) => ({ value: c.classId, label: c.name }))

  const { data, isLoading, isError, refetch } = useRunEvaluationErrors(run.id, page, classId, true)

  const errors = data?.errors ?? []
  const total = data?.total ?? 0
  const perPage = data?.perPage ?? 50
  const totalPages = Math.max(1, Math.ceil(total / perPage))

  return (
    <Card withBorder p="lg" radius="md">
      <Stack gap="md">
        <Group justify="space-between">
          <Group gap="sm">
            <Button variant="subtle" color="gray" size="xs" leftSection={<ArrowLeftIcon size={14} />} onClick={onBack}>
              Back to evaluation
            </Button>
            <Title order={5}>Misclassified items</Title>
          </Group>
          {classOptions.length > 0 && (
            <Select
              placeholder="Filter by actual class"
              size="xs"
              w={220}
              data={classOptions}
              value={classId ?? null}
              onChange={(v) => {
                setClassId(v ?? undefined)
                setPage(1)
              }}
              clearable
            />
          )}
        </Group>

        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {errors.length === 0 ? (
            <EmptyState title="No misclassified items" description="This run got every evaluated row right." compact />
          ) : (
            <Stack gap="xs">
              {errors.map((row) => (
                <ErrorCard key={row.itemId} row={row} modality={modality} />
              ))}
            </Stack>
          )}
        </QueryBoundary>

        {totalPages > 1 && (
          <Group justify="center">
            <Pagination total={totalPages} value={page} onChange={setPage} size="sm" />
          </Group>
        )}
      </Stack>
    </Card>
  )
}

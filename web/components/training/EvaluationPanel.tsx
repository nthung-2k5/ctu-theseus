/**
 * EvaluationPanel – confusion matrix, per-class stats, and regression
 * metrics for a finished run, plus an entry point into the misclassified-
 * item grid (ErrorAnalysisGrid). This is what closes the loop from "the
 * model is 87% accurate" back to "here are the items it got wrong" — see
 * server/routes/training.ts's GET /runs/:runId/evaluation(/errors).
 *
 * Axis/class labels always come from `report.idx2str` (Ludwig's own
 * class-index order, read off training_set_metadata.json server-side) —
 * never from the project's label_classes list, which has no idea which
 * index Ludwig assigned to which class. See
 * ai_service/services/evaluate.py's module docstring.
 */

import {
  Badge,
  Button,
  Card,
  Group,
  ScrollArea,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { ChartBarIcon, MagnifyingGlassIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { DataTable, type DataTableColumn, EmptyState } from '@public/components/ui'
import { useRunEvaluation } from '@public/lib/queries'
import type { EvaluationReport, Modality, PerClassStats, RunEvaluation, TrainingRunSummary } from '@public/store/types'
import { Fragment, useState } from 'react'
import { ErrorAnalysisGrid } from './ErrorAnalysisGrid'

/** Above this class count the heatmap grid stops being readable — the per-class table below still covers it. */
const MAX_MATRIX_DISPLAY = 20

const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  macroF1: 'Macro F1',
  mae: 'MAE',
  rmse: 'RMSE',
  r2: 'R²',
  loss: 'Loss',
}

function ConfusionMatrix({ report }: { report: EvaluationReport }) {
  const idx2str = report.idx2str ?? []
  const matrix = report.confusionMatrix
  if (!matrix || idx2str.length === 0) return null

  const maxValue = Math.max(1, ...matrix.flat())

  return (
    <ScrollArea>
      <div
        style={{
          display: 'inline-grid',
          gridTemplateColumns: `auto repeat(${idx2str.length}, minmax(2.25rem, 1fr))`,
          gap: 3,
          alignItems: 'center',
        }}
      >
        <div />
        {idx2str.map((label) => (
          <Text key={`col-${label}`} size="xs" c="dimmed" ta="center" title={label} truncate maw={44}>
            {label}
          </Text>
        ))}
        {matrix.map((row, i) => (
          <Fragment key={`row-${idx2str[i] ?? i}`}>
            <Text size="xs" c="dimmed" ta="right" pr="xs" title={idx2str[i]} truncate maw={100}>
              {idx2str[i]}
            </Text>
            {row.map((value, j) => {
              const intensity = value / maxValue
              const isDiagonal = i === j
              return (
                <Tooltip
                  key={`cell-${idx2str[i]}-${idx2str[j]}`}
                  label={`actual ${idx2str[i]} → predicted ${idx2str[j]}: ${value}`}
                >
                  <div
                    style={{
                      aspectRatio: '1',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: 4,
                      fontSize: 11,
                      backgroundColor: isDiagonal
                        ? `rgba(34, 139, 87, ${0.15 + intensity * 0.7})`
                        : value > 0
                          ? `rgba(224, 49, 49, ${0.1 + intensity * 0.6})`
                          : 'var(--mantine-color-default-hover)',
                    }}
                  >
                    {value > 0 ? value : ''}
                  </div>
                </Tooltip>
              )
            })}
          </Fragment>
        ))}
      </div>
    </ScrollArea>
  )
}

function PerClassTable({ perClass }: { perClass: Record<string, PerClassStats> }) {
  const rows = Object.entries(perClass).map(([label, stats]) => ({ label, ...stats }))
  const columns: DataTableColumn<(typeof rows)[number]>[] = [
    {
      key: 'label',
      header: 'Class',
      render: (r) => (
        <Text size="sm" fw={600}>
          {r.label}
        </Text>
      ),
    },
    { key: 'precision', header: 'Precision', fit: true, render: (r) => r.precision.toFixed(3) },
    { key: 'recall', header: 'Recall', fit: true, render: (r) => r.recall.toFixed(3) },
    { key: 'f1', header: 'F1', fit: true, render: (r) => r.f1.toFixed(3) },
    { key: 'support', header: 'Support', fit: true, render: (r) => r.support },
  ]
  return <DataTable columns={columns} data={rows} getRowKey={(r) => r.label} />
}

export function EvaluationPanel({
  run,
  projectId,
  modality,
}: {
  run: TrainingRunSummary
  projectId: string
  modality: Modality | undefined
}) {
  const isActive = run.status === 'running' || run.status === 'queued'
  const { data, isLoading } = useRunEvaluation(run.id, isActive)
  const [showErrors, setShowErrors] = useState(false)

  if (run.status !== 'succeeded') {
    return (
      <Card withBorder p="lg" radius="md">
        <EmptyState
          icon={ChartBarIcon}
          title="Nothing to evaluate yet"
          description="Available once the run succeeds."
        />
      </Card>
    )
  }

  if (isLoading) {
    return (
      <Card withBorder p="lg" radius="md">
        <Stack gap="md">
          <Skeleton height={24} width={200} />
          <Skeleton height={160} />
        </Stack>
      </Card>
    )
  }

  const evaluation = data?.evaluation as RunEvaluation | undefined

  if (!evaluation) {
    return (
      <Card withBorder p="lg" radius="md">
        <EmptyState
          icon={ChartBarIcon}
          title="No evaluation report"
          description="This run has no evaluation report — it may have been trained before this feature was added."
        />
      </Card>
    )
  }

  if (evaluation.status === 'failed') {
    return (
      <Card withBorder p="lg" radius="md">
        <Group gap="xs" mb="xs">
          <WarningCircleIcon size={16} color="var(--mantine-color-red-6)" />
          <Text size="sm" fw={600} c="red">
            Evaluation failed
          </Text>
        </Group>
        <Text size="sm" c="dimmed">
          {evaluation.failedMessage ?? 'The worker could not produce an evaluation report for this run.'}
        </Text>
      </Card>
    )
  }

  const report = evaluation.report
  if (!report) return null

  if (showErrors) {
    return (
      <ErrorAnalysisGrid
        run={run}
        projectId={projectId}
        modality={modality}
        classNames={report.idx2str ?? []}
        onBack={() => setShowErrors(false)}
      />
    )
  }

  const overallEntries = Object.entries(report.overall).filter(([, value]) => value !== null) as [string, number][]
  const showMatrix = !!report.confusionMatrix && (report.idx2str?.length ?? 0) <= MAX_MATRIX_DISPLAY

  return (
    <Card withBorder p="lg" radius="md">
      <Stack gap="lg">
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={5}>Evaluation</Title>
            <Text size="xs" c="dimmed">
              Evaluated on the {report.split} split · {report.rowCount} row{report.rowCount === 1 ? '' : 's'}
            </Text>
          </div>
          {report.truncated && (
            <Badge color="yellow" variant="light">
              {report.idx2str?.length ?? 0} classes — matrix hidden
            </Badge>
          )}
        </Group>

        {overallEntries.length > 0 && (
          <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md">
            {overallEntries.map(([key, value]) => (
              <div key={key}>
                <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                  {METRIC_LABELS[key] ?? key}
                </Text>
                <Text size="lg" fw={700}>
                  {value.toFixed(3)}
                </Text>
              </div>
            ))}
          </SimpleGrid>
        )}

        {showMatrix && (
          <div>
            <Text size="sm" fw={600} mb={2}>
              Confusion Matrix
            </Text>
            <Text size="xs" c="dimmed" mb="xs">
              Rows are the actual class, columns are the predicted class.
            </Text>
            <ConfusionMatrix report={report} />
          </div>
        )}

        {report.perClass && (
          <div>
            <Text size="sm" fw={600} mb="xs">
              Per-class metrics
            </Text>
            <PerClassTable perClass={report.perClass} />
          </div>
        )}

        {report.topErrors && report.topErrors.length > 0 && (
          <Group justify="flex-end">
            <Button variant="light" leftSection={<MagnifyingGlassIcon size={14} />} onClick={() => setShowErrors(true)}>
              View {report.topErrors.length} misclassified item{report.topErrors.length === 1 ? '' : 's'}
            </Button>
          </Group>
        )}
      </Stack>
    </Card>
  )
}

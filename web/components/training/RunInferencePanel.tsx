/**
 * Playground input and result for one training run: pick an input, send it, and show the parsed
 * prediction. The prediction runs inside the request and its result is the response, so there is
 * nothing to poll and nothing saved on the server.
 */

import {
  ActionIcon,
  Alert,
  Box,
  Button,
  Card,
  Grid,
  Group,
  JsonInput,
  Loader,
  Overlay,
  Paper,
  Stack,
  Text,
  Textarea,
  ThemeIcon,
  Title,
  Tooltip,
  Transition,
} from '@mantine/core'
import { Dropzone, type FileRejection } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import {
  ArrowCounterClockwiseIcon,
  CheckCircleIcon,
  CloudArrowUpIcon,
  LightningIcon,
  XCircleIcon,
} from '@phosphor-icons/react'
import { EmptyState } from '@public/components/ui'
import { apiErrorMessage, axios } from '@public/lib/api/client'
import { runInference, warmInferenceModel } from '@public/lib/api/generated/inference/inference'
import { projectDetailQueryOptions } from '@public/lib/queries'
import { getInferenceInputSpec, getTaskDescriptor } from '@public/lib/tasks'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useSuspenseQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { type InferenceOutput, InferenceResultStats } from './InferenceResultStats'

/** What the last request produced. A batch result is a scored CSV held in the browser, never on the server. */
type Result = { kind: 'single'; output: InferenceOutput } | { kind: 'batch'; rowCount: number; url: string }

/** Score a CSV. The response is the scored file itself; its row count is in a header. */
async function scoreBatch(runId: string, file: File): Promise<{ rowCount: number; url: string }> {
  const body = new FormData()
  body.append('file', file)
  try {
    const res = await axios.post<Blob>(`/api/inference/${runId}/batch`, body, { responseType: 'blob' })
    return { rowCount: Number(res.headers['x-row-count'] ?? 0), url: URL.createObjectURL(res.data) }
  } catch (error) {
    // Error bodies arrive as a Blob too (responseType applies to every response): read the JSON message out of it.
    const blob = (error as { response?: { data?: unknown } }).response?.data
    if (blob instanceof Blob) {
      try {
        const message = JSON.parse(await blob.text())?.error?.message
        if (typeof message === 'string') throw new Error(message)
      } catch (parsed) {
        if (parsed instanceof Error && !(parsed instanceof SyntaxError)) throw parsed
      }
    }
    throw new Error(apiErrorMessage(error, 'Could not run inference'))
  }
}

function fieldLabel(field: string): string {
  return field.charAt(0).toUpperCase() + field.slice(1)
}

export function RunInferencePanel({ projectId, run }: { projectId: string; run: TrainingRunSummary }) {
  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const inputSpec = getInferenceInputSpec(activeProject.task)
  const isAudio = getTaskDescriptor(activeProject.task).modality === 'audio'

  const runId = run.id
  const isReady = run.status === 'succeeded'

  // File input (vision/audio tasks)
  const [inputFile, setInputFile] = useState<File | null>(null)
  const [inputFileUrl, setInputFileUrl] = useState<string | null>(null)
  // Text input (text tasks) — one value per Ludwig input field, e.g.
  // question_answering needs both `context` and `question`.
  const [textFields, setTextFields] = useState<Record<string, string>>({})
  // Record input (tabular tasks) — JSON-encoded, one value per feature column
  const [recordJson, setRecordJson] = useState('')

  // The last request's result, cleared by handleClear or a fresh input.
  const [result, setResult] = useState<Result | null>(null)

  // Confidence threshold — only meaningful for classification outputs; the
  // server returns the full (top-K) distribution unfiltered, so moving
  // this slider can freely reveal classes it initially hid.
  const [threshold, setThreshold] = useState(0.5)

  const isFileTask = inputSpec.kind === 'file'
  const isTextTask = inputSpec.kind === 'text'
  const isRecordTask = inputSpec.kind === 'record'

  // Batch mode: score every row of an uploaded CSV in one request instead of
  // one row at a time — only meaningful for text/tabular tasks (see
  // server/routes/inference.ts's POST /inference/:runId/batch).
  const [batchMode, setBatchMode] = useState(false)
  const [batchFile, setBatchFile] = useState<File | null>(null)

  const hasInput = batchMode
    ? !!batchFile
    : isFileTask
      ? !!inputFile
      : isTextTask
        ? inputSpec.fields.every((f) => (textFields[f] ?? '').trim().length > 0)
        : recordJson.trim().length > 0

  // Preload the selected run's model into the worker's cache ahead of the
  // user's first request — closes the cold-start gap the model cache alone
  // can't help with (it only speeds up the *second* request for a run).
  // Best-effort: a failure here just means the first real request pays the
  // normal cold-start cost instead.
  useEffect(() => {
    if (!isReady) return
    warmInferenceModel(runId).catch(() => {})
  }, [runId, isReady])

  // Switching runs keeps this panel mounted — drop the result belonging to the run we just left.
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional reset-on-change, runId isn't read in the body
  useEffect(() => {
    setResult(null)
  }, [runId])

  // A batch result is an object URL: free it when replaced or unmounted.
  const batchUrl = result?.kind === 'batch' ? result.url : null
  useEffect(() => {
    if (!batchUrl) return
    return () => URL.revokeObjectURL(batchUrl)
  }, [batchUrl])

  /* ── Run ───────────────────────────────────────────────────────── */
  // The prediction happens inside this request: the response is the result. Nothing is queued or saved.
  const dispatchMutation = useMutation({
    mutationFn: async (): Promise<Result> => {
      try {
        if (batchMode) return { kind: 'batch', ...(await scoreBatch(runId, batchFile as File)) }
        const { output } = await runInference(
          runId,
          isFileTask
            ? { file: inputFile as File }
            : isTextTask
              ? { fields: JSON.stringify(textFields) }
              : { fields: recordJson },
        )
        return { kind: 'single', output: output as InferenceOutput }
      } catch (error) {
        throw new Error(apiErrorMessage(error, 'Could not run inference'))
      }
    },
    onSuccess: (next) => {
      setResult(next)
      if (next.kind === 'batch') {
        notifications.show({
          title: 'Batch complete',
          message: `${next.rowCount} row(s) scored: download the results below`,
          color: 'teal',
          icon: <CheckCircleIcon size={18} />,
        })
      }
    },
  })

  /* ── Handlers ──────────────────────────────────────────────────── */
  // Object URLs are not garbage-collected with the File they wrap — without
  // this every file the user tests stays pinned for the lifetime of the tab,
  // and the dropzone accepts up to 25 MB a time.
  useEffect(() => {
    if (!inputFileUrl) return
    return () => URL.revokeObjectURL(inputFileUrl)
  }, [inputFileUrl])

  const handleDrop = (files: File[]) => {
    if (files.length === 0) return
    const file = files[0]
    setInputFile(file)
    setInputFileUrl(URL.createObjectURL(file))
    setResult(null)
  }

  const handleReject = (rejections: FileRejection[]) => {
    const reason = rejections[0]?.errors[0]?.message ?? 'File was rejected'
    notifications.show({ title: 'File not accepted', message: reason, color: 'orange' })
  }

  const handleBatchDrop = (files: File[]) => {
    if (files.length === 0) return
    setBatchFile(files[0])
    setResult(null)
  }

  const handleClear = () => {
    setInputFileUrl(null)
    setInputFile(null)
    setTextFields({})
    setRecordJson('')
    setBatchFile(null)
    setResult(null)
    dispatchMutation.reset()
  }

  const isRunning = dispatchMutation.isPending
  const output = result?.kind === 'single' ? result.output : null
  const batchRowCount = result?.kind === 'batch' ? result.rowCount : null

  const dispatchError = dispatchMutation.error as Error | null
  const displayError = dispatchError ? { title: 'Inference failed', message: dispatchError.message } : null

  /* ── Render ────────────────────────────────────────────────────── */
  if (!isReady) {
    return (
      <Card withBorder p="lg" radius="md">
        <EmptyState
          icon={LightningIcon}
          title="Nothing to test yet"
          description="This run has to finish successfully before you can run predictions with it."
        />
      </Card>
    )
  }

  return (
    <Grid gap="md">
      {/* ── Input ── */}
      <Grid.Col span={{ base: 12, md: 6 }}>
        <Stack gap="md">
          {/* Input */}
          <Card withBorder padding="lg" radius="md">
            <Group justify="space-between" mb="md">
              <Title order={5}>
                {batchMode ? 'Batch Input' : isFileTask ? (isAudio ? 'Test Audio' : 'Test Image') : 'Test Input'}
              </Title>
              <Group gap="xs">
                {!isFileTask && (
                  <Button
                    size="compact-xs"
                    variant={batchMode ? 'filled' : 'default'}
                    onClick={() => {
                      setBatchMode((prev) => !prev)
                      handleClear()
                    }}
                  >
                    Batch
                  </Button>
                )}
                {hasInput && (
                  <Tooltip label="Clear">
                    <ActionIcon variant="subtle" color="gray" onClick={handleClear}>
                      <ArrowCounterClockwiseIcon size={18} />
                    </ActionIcon>
                  </Tooltip>
                )}
              </Group>
            </Group>

            {batchMode && !batchFile && (
              <Dropzone
                onDrop={handleBatchDrop}
                onReject={handleReject}
                accept={['text/csv']}
                maxSize={25 * 1024 * 1024}
                radius="md"
                p="xl"
                style={{ borderStyle: 'dashed', borderWidth: 2, transition: 'all 150ms ease' }}
              >
                <Stack align="center" gap="sm">
                  <ThemeIcon size={56} variant="light" color="primary" radius="xl">
                    <CloudArrowUpIcon size={30} />
                  </ThemeIcon>
                  <Text size="sm" fw={500}>
                    Drop a CSV here or click to browse
                  </Text>
                  <Text size="xs" c="dimmed">
                    One column per input field, one row per prediction — up to 10,000 rows
                  </Text>
                </Stack>
              </Dropzone>
            )}

            {batchMode && batchFile && (
              <Paper withBorder p="md" radius="md">
                <Group justify="space-between">
                  <Text size="sm">{batchFile.name}</Text>
                  <Text size="xs" c="dimmed">
                    {(batchFile.size / 1024).toFixed(1)} KB
                  </Text>
                </Group>
              </Paper>
            )}

            {!batchMode && isFileTask && !inputFileUrl && (
              <Dropzone
                onDrop={handleDrop}
                onReject={handleReject}
                accept={inputSpec.accept}
                maxSize={25 * 1024 * 1024}
                radius="md"
                p="xl"
                style={{ borderStyle: 'dashed', borderWidth: 2, transition: 'all 150ms ease' }}
              >
                <Stack align="center" gap="sm">
                  <ThemeIcon size={56} variant="light" color="primary" radius="xl">
                    <CloudArrowUpIcon size={30} />
                  </ThemeIcon>
                  <Text size="sm" fw={500}>
                    Drop {isAudio ? 'an audio file' : 'an image'} here or click to browse
                  </Text>
                </Stack>
              </Dropzone>
            )}

            {isFileTask && inputFileUrl && (
              <Box pos="relative">
                <Paper radius="md" style={{ overflow: 'hidden', position: 'relative' }}>
                  {isAudio ? (
                    // biome-ignore lint/a11y/useMediaCaption: user-uploaded test clip, no transcript source
                    <audio controls src={inputFileUrl} style={{ width: '100%' }} />
                  ) : (
                    <img
                      src={inputFileUrl}
                      alt="Test"
                      style={{
                        width: '100%',
                        display: 'block',
                        transition: 'filter 300ms ease',
                        filter: isRunning ? 'brightness(0.6)' : 'none',
                      }}
                    />
                  )}
                  {isRunning && (
                    <Overlay center backgroundOpacity={0.4} blur={2} radius="md">
                      <Stack align="center" gap="xs">
                        <Loader size="lg" color="white" />
                        <Text size="sm" c="white" fw={500}>
                          Running inference...
                        </Text>
                      </Stack>
                    </Overlay>
                  )}
                </Paper>
              </Box>
            )}

            {!batchMode && isTextTask && (
              <Stack gap="sm" pos="relative">
                {inputSpec.fields.map((field) => (
                  <Textarea
                    key={field}
                    label={inputSpec.fields.length > 1 ? fieldLabel(field) : undefined}
                    placeholder={`Paste or type ${field}`}
                    autosize
                    minRows={4}
                    value={textFields[field] ?? ''}
                    onChange={(e) => {
                      setTextFields((prev) => ({ ...prev, [field]: e.currentTarget.value }))
                      setResult(null)
                    }}
                  />
                ))}
                {isRunning && (
                  <Overlay center backgroundOpacity={0.15} radius="md">
                    <Loader size="sm" />
                  </Overlay>
                )}
              </Stack>
            )}

            {!batchMode && isRecordTask && (
              <Box pos="relative">
                <JsonInput
                  placeholder='{"age": 34, "income": 52000}'
                  description="One JSON object with a value for each feature column"
                  autosize
                  minRows={4}
                  formatOnBlur
                  value={recordJson}
                  onChange={(v) => {
                    setRecordJson(v)
                    setResult(null)
                  }}
                />
                {isRunning && (
                  <Overlay center backgroundOpacity={0.15} radius="md">
                    <Loader size="sm" />
                  </Overlay>
                )}
              </Box>
            )}

            {hasInput && (
              <Group mt="md" gap="sm">
                <Button
                  onClick={() => dispatchMutation.mutate()}
                  loading={isRunning}
                  leftSection={<LightningIcon size={18} weight="fill" />}
                  variant="gradient"
                  gradient={{ from: 'primary', to: 'secondary' }}
                >
                  {batchMode ? 'Run Batch' : 'Run Inference'}
                </Button>
                <Button variant="subtle" color="gray" onClick={handleClear}>
                  Clear
                </Button>
              </Group>
            )}
          </Card>

          {/* Error */}
          {displayError && (
            <Alert
              color="red"
              icon={<XCircleIcon size={18} />}
              title={displayError.title}
              withCloseButton
              onClose={handleClear}
            >
              {displayError.message}
            </Alert>
          )}
        </Stack>
      </Grid.Col>

      {/* ── Results ── */}
      <Grid.Col span={{ base: 12, md: 6 }}>
        <Stack gap="md">
          {!output && batchRowCount === null && !isRunning && !displayError && (
            <Paper p="xl" ta="center">
              <Text size="sm" c="dimmed">
                Results appear here once you run a prediction.
              </Text>
            </Paper>
          )}

          {/* Batch result — a downloadable file, not an inline InferenceOutput */}
          <Transition mounted={batchRowCount !== null} transition="slide-up" duration={300}>
            {(styles) => (
              <Card withBorder padding="lg" radius="md" style={styles}>
                <Group justify="space-between">
                  <Group gap="sm">
                    <ThemeIcon variant="light" color="teal" size="sm">
                      <CheckCircleIcon size={14} />
                    </ThemeIcon>
                    <div>
                      <Title order={5}>Batch Complete</Title>
                      <Text size="xs" c="dimmed">
                        {batchRowCount ?? 0} row(s) scored
                      </Text>
                    </div>
                  </Group>
                  {result?.kind === 'batch' && (
                    <Button component="a" href={result.url} download="predictions.csv" variant="light" color="teal">
                      Download Results
                    </Button>
                  )}
                </Group>
              </Card>
            )}
          </Transition>

          {/* Parsed result: prediction, confidence bars and per-kind statistics (no raw JSON) */}
          <Transition mounted={!!output} transition="slide-up" duration={300}>
            {(styles) => (
              <div style={styles}>
                {output && (
                  <InferenceResultStats output={output} threshold={threshold} onThresholdChange={setThreshold} />
                )}
              </div>
            )}
          </Transition>
        </Stack>
      </Grid.Col>
    </Grid>
  )
}

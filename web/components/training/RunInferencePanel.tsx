/**
 * Inference tab for a selected training run — was the standalone Inference
 * page, which carried its own "pick a trained model" dropdown. The run now
 * comes from the Training sidebar, so this only handles the input -> dispatch
 * -> poll -> render-result loop for that one run.
 */

import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Code,
  CopyButton,
  Divider,
  Grid,
  Group,
  JsonInput,
  Loader,
  Overlay,
  Paper,
  Progress,
  Slider,
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
  CheckIcon,
  ClockCounterClockwiseIcon,
  CloudArrowUpIcon,
  CopyIcon,
  CrosshairIcon,
  LightningIcon,
  WarningCircleIcon,
  XCircleIcon,
} from '@phosphor-icons/react'
import { EmptyState } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  getInferenceJob,
  listInferenceJobs,
  runBatchInference,
  runInference,
  warmInferenceModel,
} from '@public/lib/api/generated/inference/inference'
import { projectDetailQueryOptions } from '@public/lib/queries'
import { getInferenceInputSpec, getInferenceOutputKind, getTaskDescriptor } from '@public/lib/tasks'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

/* ------------------------------------------------------------------ */
/*  Types — mirrors InferenceOutputSchema in server/lib/schema.ts.     */
/*  Hand-maintained rather than Eden-derived, same as web/store/types  */
/*  (see README.md's note on the server/web Elysia nominal-typing gap). */
/* ------------------------------------------------------------------ */

type InferenceOutput =
  | { kind: 'classification'; feature: string; classes: { label: string; confidence: number }[] }
  | { kind: 'regression'; feature: string; value: number }
  | { kind: 'text'; feature: string; text: string }
  | { kind: 'tokens'; feature: string; tokens: { token: string; tag: string }[] }

/**
 * The poll route's response — `pending` is synthesized by the gateway when
 * no terminal result has been published yet for the job, so unlike the
 * dispatch-time payload this genuinely has three states, not two. `batch`
 * carries no inline output (see server/routes/inference.ts) — only a row
 * count; the actual CSV is fetched via the download route.
 */
type InferenceJobStatus =
  | { status: 'pending' }
  | { status: 'success'; output: InferenceOutput }
  | { status: 'batch'; rowCount: number }
  | { status: 'failed'; error: string }

/** One row of GET /inference/:runId/jobs — mirrors the inferenceJobs table (server/db/schema.ts). */
interface InferenceJobHistoryItem {
  id: string
  runId: string
  status: 'pending' | 'success' | 'failed'
  output: (InferenceOutput | { kind: 'batch'; resultKey: string; rowCount: number }) | null
  error: string | null
  createdAt: string | Date
  completedAt: string | Date | null
}

/** How long to keep polling a job with no result before assuming something's stuck (e.g. no worker running at all). */
const POLL_GIVE_UP_MS = 5 * 60 * 1000
const POLL_INTERVAL_MS = 1500

function describeOutput(output: InferenceOutput | { kind: 'batch'; resultKey: string; rowCount: number }): string {
  switch (output.kind) {
    case 'classification':
      return `${output.classes.length} result(s)`
    case 'regression':
      return `Predicted ${output.feature}: ${output.value}`
    case 'text':
      return 'Generated response ready'
    case 'tokens':
      return `${output.tokens.length} token(s) tagged`
    case 'batch':
      return `${output.rowCount} row(s) scored`
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
  const outputKind = getInferenceOutputKind(activeProject.task)
  const isAudio = getTaskDescriptor(activeProject.task).modality === 'audio'

  const runId = run.id
  const isReady = run.status === 'succeeded'
  const queryClient = useQueryClient()

  /* ── Inference history — persisted server-side the moment a result arrives, not just while polling ── */
  const historyQueryKey = ['inference-jobs', runId]
  const historyQuery = useQuery({
    queryKey: historyQueryKey,
    queryFn: async (): Promise<InferenceJobHistoryItem[]> => {
      return (await listInferenceJobs(runId)).jobs as InferenceJobHistoryItem[]
    },
    enabled: isReady,
  })
  const history = historyQuery.data ?? []

  // File input (vision/audio tasks)
  const [inputFile, setInputFile] = useState<File | null>(null)
  const [inputFileUrl, setInputFileUrl] = useState<string | null>(null)
  // Text input (text tasks) — one value per Ludwig input field, e.g.
  // question_answering needs both `context` and `question`.
  const [textFields, setTextFields] = useState<Record<string, string>>({})
  // Record input (tabular tasks) — JSON-encoded, one value per feature column
  const [recordJson, setRecordJson] = useState('')

  // The currently-dispatched job, if any — set once POST /inference/:runId
  // returns 202 { inferenceId }, cleared by handleClear or a fresh dispatch.
  const [inferenceId, setInferenceId] = useState<string | null>(null)
  const [gaveUp, setGaveUp] = useState(false)

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

  // Switching runs in the sidebar keeps this panel mounted — drop any job
  // state belonging to the run we just navigated away from.
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional reset-on-change, runId isn't read in the body
  useEffect(() => {
    setInferenceId(null)
    setGaveUp(false)
  }, [runId])

  /* ── Dispatch ──────────────────────────────────────────────────── */
  const dispatchMutation = useMutation({
    mutationFn: async (): Promise<string> => {
      try {
        const accepted = batchMode
          ? await runBatchInference(runId, { file: batchFile as File })
          : await runInference(
              runId,
              isFileTask
                ? { file: inputFile as File }
                : isTextTask
                  ? { fields: JSON.stringify(textFields) }
                  : { fields: recordJson },
            )
        return accepted.inferenceId
      } catch (error) {
        throw new Error(apiErrorMessage(error, 'Could not start inference'))
      }
    },
    onSuccess: (id) => {
      setInferenceId(id)
      setGaveUp(false)
    },
  })

  /* ── Poll ──────────────────────────────────────────────────────── */
  const jobQuery = useQuery({
    queryKey: ['inference-job', runId, inferenceId],
    queryFn: async (): Promise<InferenceJobStatus> => {
      return (await getInferenceJob(runId, inferenceId as string)) as InferenceJobStatus
    },
    enabled: !!inferenceId,
    refetchInterval: (query) => (query.state.data?.status === 'pending' && !gaveUp ? POLL_INTERVAL_MS : false),
  })

  const job = jobQuery.data

  // Give up auto-polling after a while rather than spinning forever if a
  // job's message never reaches a worker at all (the one failure mode the
  // worker's own retry/DLQ machinery can't resolve into a terminal result).
  useEffect(() => {
    if (!inferenceId || job?.status !== 'pending') return
    const timer = setTimeout(() => setGaveUp(true), POLL_GIVE_UP_MS)
    return () => clearTimeout(timer)
  }, [inferenceId, job?.status])

  // Notify once per terminal result — `job` only gets a new object identity
  // when its content actually changes, and refetchInterval stops once a
  // job reaches success/failed, so this can't re-fire for the same result.
  const notifiedFor = useRef<string | null>(null)
  useEffect(() => {
    if (!inferenceId || !job || job.status === 'pending' || notifiedFor.current === inferenceId) return
    notifiedFor.current = inferenceId
    queryClient.invalidateQueries({ queryKey: ['inference-jobs', runId] })
    if (job.status === 'success') {
      notifications.show({
        title: 'Inference Complete',
        message: describeOutput(job.output),
        color: 'teal',
        icon: <CheckCircleIcon size={18} />,
      })
    } else if (job.status === 'batch') {
      notifications.show({
        title: 'Batch Complete',
        message: `${job.rowCount} row(s) scored — download the results below`,
        color: 'teal',
        icon: <CheckCircleIcon size={18} />,
      })
    } else {
      notifications.show({
        title: 'Inference Failed',
        message: job.error,
        color: 'red',
        icon: <XCircleIcon size={18} />,
      })
    }
  }, [inferenceId, job, runId, queryClient])

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
    setInferenceId(null)
  }

  const handleReject = (rejections: FileRejection[]) => {
    const reason = rejections[0]?.errors[0]?.message ?? 'File was rejected'
    notifications.show({ title: 'File not accepted', message: reason, color: 'orange' })
  }

  const handleBatchDrop = (files: File[]) => {
    if (files.length === 0) return
    setBatchFile(files[0])
    setInferenceId(null)
  }

  const handleClear = () => {
    setInputFileUrl(null)
    setInputFile(null)
    setTextFields({})
    setRecordJson('')
    setBatchFile(null)
    setInferenceId(null)
    setGaveUp(false)
    dispatchMutation.reset()
  }

  const isRunning = dispatchMutation.isPending || (!!inferenceId && job?.status === 'pending' && !gaveUp)
  const output = job?.status === 'success' ? job.output : null
  const batchRowCount = job?.status === 'batch' ? job.rowCount : null

  // Classification results only: client-side threshold filter + sort. The
  // server already sorts descending, so filtering preserves order.
  const filteredClasses =
    output?.kind === 'classification' ? output.classes.filter((c) => c.confidence >= threshold) : null

  const dispatchError = dispatchMutation.error as Error | null
  const displayError = dispatchError
    ? { title: 'Could not start inference', message: dispatchError.message }
    : job?.status === 'failed'
      ? { title: 'Inference failed', message: job.error }
      : null

  const showThreshold = outputKind === 'classification' && !batchMode

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
      {/* ── Input & Results ── */}
      <Grid.Col span={{ base: 12, md: showThreshold ? 8 : 12 }}>
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
                      setInferenceId(null)
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
                    setInferenceId(null)
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

          {/* Stuck-job warning */}
          {gaveUp && job?.status === 'pending' && (
            <Alert
              color="yellow"
              icon={<WarningCircleIcon size={18} />}
              title="Still waiting"
              withCloseButton
              onClose={handleClear}
            >
              This is taking much longer than usual — the worker may be unavailable right now. You can keep waiting or
              clear and try again.
              <Group mt="xs">
                <Button size="xs" variant="light" onClick={() => setGaveUp(false)}>
                  Keep waiting
                </Button>
              </Group>
            </Alert>
          )}

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
                  {inferenceId && (
                    <Button
                      component="a"
                      href={`/api/inference/${runId}/jobs/${inferenceId}/download`}
                      variant="light"
                      color="teal"
                    >
                      Download Results
                    </Button>
                  )}
                </Group>
              </Card>
            )}
          </Transition>

          {/* Results */}
          <Transition mounted={!!output} transition="slide-up" duration={300}>
            {(styles) => (
              <Card withBorder padding="lg" radius="md" style={styles}>
                <Group justify="space-between" mb="md">
                  <Group gap="sm">
                    <ThemeIcon variant="light" color="teal" size="sm">
                      <CrosshairIcon size={14} />
                    </ThemeIcon>
                    <Title order={5}>Results</Title>
                  </Group>
                  {output?.kind === 'classification' && (
                    <Badge variant="light" color="teal" size="lg">
                      {filteredClasses?.length ?? 0} result(s)
                    </Badge>
                  )}
                </Group>

                {output?.kind === 'classification' &&
                  (filteredClasses && filteredClasses.length > 0 ? (
                    <Stack gap="xs">
                      {filteredClasses.map((c) => (
                        <Paper key={c.label} p="sm" radius="sm" withBorder>
                          <Group justify="space-between">
                            <Badge variant="filled" color="primary" size="sm">
                              {c.label}
                            </Badge>
                            <Group gap="xs">
                              <Progress
                                value={c.confidence * 100}
                                color={c.confidence > 0.8 ? 'teal' : c.confidence > 0.5 ? 'yellow' : 'red'}
                                size="sm"
                                w={80}
                              />
                              <Text size="xs" fw={600} w={45} ta="right">
                                {(c.confidence * 100).toFixed(1)}%
                              </Text>
                            </Group>
                          </Group>
                        </Paper>
                      ))}
                      <Divider my="xs" />
                      <Code block style={{ maxHeight: 200, overflow: 'auto' }}>
                        {JSON.stringify(output.classes, null, 2)}
                      </Code>
                    </Stack>
                  ) : (
                    <Text size="sm" c="dimmed" ta="center" py="md">
                      No results above the confidence threshold
                    </Text>
                  ))}

                {output?.kind === 'regression' && (
                  <Paper p="lg" radius="sm" withBorder ta="center">
                    <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                      {output.feature}
                    </Text>
                    <Text size="xl" fw={700}>
                      {output.value}
                    </Text>
                  </Paper>
                )}

                {output?.kind === 'text' && (
                  <Paper p="md" radius="sm" withBorder pos="relative">
                    <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                      {output.text}
                    </Text>
                    <CopyButton value={output.text}>
                      {({ copied, copy }) => (
                        <ActionIcon
                          variant="subtle"
                          color={copied ? 'teal' : 'gray'}
                          onClick={copy}
                          pos="absolute"
                          top={8}
                          right={8}
                        >
                          {copied ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                        </ActionIcon>
                      )}
                    </CopyButton>
                  </Paper>
                )}

                {output?.kind === 'tokens' && (
                  <Group gap="xs">
                    {output.tokens.map((t, i) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: token/tag pairs have no stable identity
                      <Tooltip key={i} label={t.tag}>
                        <Badge variant="light" color={t.tag === 'O' ? 'gray' : 'primary'} size="lg">
                          {t.token || '·'}
                        </Badge>
                      </Tooltip>
                    ))}
                  </Group>
                )}
              </Card>
            )}
          </Transition>

          {/* Recent jobs — persisted server-side the instant a result arrives, so past results are still viewable */}
          {history.length > 0 && (
            <Card withBorder padding="lg" radius="md">
              <Group gap="sm" mb="sm">
                <ClockCounterClockwiseIcon size={18} />
                <Title order={5}>Recent Jobs</Title>
              </Group>
              <Stack gap={4}>
                {history.map((item) => (
                  <Group
                    key={item.id}
                    justify="space-between"
                    p="xs"
                    style={{ cursor: 'pointer', borderRadius: 6 }}
                    onClick={() => setInferenceId(item.id)}
                  >
                    <Group gap="sm">
                      {item.status === 'success' ? (
                        <CheckCircleIcon size={16} color="var(--mantine-color-teal-6)" />
                      ) : item.status === 'failed' ? (
                        <XCircleIcon size={16} color="var(--mantine-color-red-6)" />
                      ) : (
                        <Loader size={14} />
                      )}
                      <Text size="xs" c="dimmed">
                        {new Date(item.createdAt).toLocaleString()}
                      </Text>
                    </Group>
                    <Text size="xs" c="dimmed" truncate maw={220}>
                      {item.status === 'success' && item.output
                        ? describeOutput(item.output)
                        : item.status === 'failed'
                          ? (item.error ?? 'Failed')
                          : 'Pending'}
                    </Text>
                  </Group>
                ))}
              </Stack>
            </Card>
          )}
        </Stack>
      </Grid.Col>

      {/* ── Confidence threshold — only meaningful for classification outputs ── */}
      {showThreshold && (
        <Grid.Col span={{ base: 12, md: 4 }}>
          <Card withBorder padding="lg" radius="md">
            <Title order={5} mb="md">
              Confidence Threshold
            </Title>
            <Text size="sm" c="dimmed" mb="sm">
              Filter out results below this confidence score
            </Text>
            <Slider
              value={threshold}
              onChange={setThreshold}
              min={0}
              max={1}
              step={0.01}
              label={(v) => `${(v * 100).toFixed(0)}%`}
              marks={[
                { value: 0.25, label: '25%' },
                { value: 0.5, label: '50%' },
                { value: 0.75, label: '75%' },
              ]}
              color="primary"
            />
            <Text size="xs" ta="center" c="dimmed" mt="md">
              Current: {(threshold * 100).toFixed(0)}%
            </Text>
          </Card>
        </Grid.Col>
      )}
    </Grid>
  )
}

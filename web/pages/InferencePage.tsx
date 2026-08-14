import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Divider,
  Grid,
  Group,
  JsonInput,
  Loader,
  Overlay,
  Paper,
  Progress,
  Select,
  Slider,
  Stack,
  Text,
  Textarea,
  ThemeIcon,
  Title,
  Tooltip,
  Transition,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import {
  ArrowCounterClockwiseIcon,
  CheckCircleIcon,
  CloudArrowUpIcon,
  CrosshairIcon,
  LightningIcon,
  XCircleIcon,
} from '@phosphor-icons/react'
import { rest } from '@public/lib/api'
import { projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import { getTaskDescriptor } from '@server/lib/tasks'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, Link } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/inference')

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

/** Raw inference result: { className: confidence } (or { value: number } for regression). */
type InferenceResult = Record<string, number>

interface DetectionEntry {
  className: string
  confidence: number
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function InferencePage() {
  const { projectId } = routeApi.useParams()
  const { runId: selectedRunId } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const setSelectedRunId = (runId: string | null) =>
    navigate({ search: (prev) => ({ ...prev, runId: runId ?? undefined }) })

  const {
    data: { project: activeProject },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const descriptor = getTaskDescriptor(activeProject.task)

  // File input (vision/audio tasks)
  const [inputFile, setInputFile] = useState<File | null>(null)
  const [inputFileUrl, setInputFileUrl] = useState<string | null>(null)
  // Text input (text tasks)
  const [textValue, setTextValue] = useState('')
  // Record input (tabular tasks) — JSON-encoded, one value per feature column
  const [recordJson, setRecordJson] = useState('')

  const [detections, setDetections] = useState<InferenceResult | null>(null)

  // Confidence threshold
  const [threshold, setThreshold] = useState(0.5)

  // Status
  const [isInferring, setIsInferring] = useState(false)

  // Fetch succeeded training runs
  const { data: runsData } = useTrainingRuns(projectId)
  const completedRuns = (runsData?.runs ?? []).filter((r) => r.status === 'succeeded')

  const isFileTask = descriptor.itemSpec.payload === 'file'
  const isTextTask = descriptor.itemSpec.payload === 'inline_text'
  const isRecordTask = descriptor.itemSpec.payload === 'record'
  const isAudio = descriptor.modality === 'audio'

  const hasInput = isFileTask ? !!inputFile : isTextTask ? textValue.trim().length > 0 : recordJson.trim().length > 0

  /* ── Handlers ──────────────────────────────────────────────────── */
  const handleDrop = (files: File[]) => {
    if (files.length === 0) return
    const file = files[0]
    setInputFile(file)
    setInputFileUrl(URL.createObjectURL(file))
    setDetections(null)
  }

  const handleInfer = async () => {
    if (!hasInput || !selectedRunId) {
      notifications.show({
        title: 'Not Ready',
        message: 'Please select a model and provide an input.',
        color: 'orange',
      })
      return
    }
    setIsInferring(true)
    setDetections(null)

    try {
      const body = isFileTask
        ? { file: inputFile as File, threshold }
        : isTextTask
          ? { text: textValue, threshold }
          : { record: recordJson, threshold }

      const response = await rest.inference({ runId: selectedRunId }).post(body)

      if (response.error) {
        notifications.show({
          title: 'Inference Failed',
          message: typeof response.error.value === 'string' ? response.error.value : 'Unknown error',
          color: 'red',
          icon: <XCircleIcon size={18} />,
        })
      } else if (response.data) {
        const data = response.data as
          | { status: 'success'; results: InferenceResult }
          | { status: 'failed'; error: string }
        if (data.status === 'success' && data.results) {
          setDetections(data.results)
          notifications.show({
            title: 'Inference Complete',
            message: `${Object.keys(data.results).length} result(s)`,
            color: 'teal',
            icon: <CheckCircleIcon size={18} />,
          })
        } else if (data.status === 'failed') {
          notifications.show({
            title: 'Inference Failed',
            message: data.error ?? 'Unknown error',
            color: 'red',
            icon: <XCircleIcon size={18} />,
          })
        }
      }
    } catch (err) {
      notifications.show({
        title: 'Inference Error',
        message: err instanceof Error ? err.message : 'Unknown error',
        color: 'red',
      })
    } finally {
      setIsInferring(false)
    }
  }

  const handleClear = () => {
    setInputFileUrl(null)
    setInputFile(null)
    setTextValue('')
    setRecordJson('')
    setDetections(null)
  }

  // Convert dict to sorted array and apply threshold filter
  const filteredDetections: DetectionEntry[] | null = detections
    ? Object.entries(detections)
        .map(([className, confidence]) => ({ className, confidence }))
        .filter((d) => d.confidence >= threshold)
        .sort((a, b) => b.confidence - a.confidence)
    : null

  /* ── Render ────────────────────────────────────────────────────── */
  return (
    <Box>
      <Stack gap="xl">
        {/* Page header */}
        <div>
          <Title order={2}>Inference</Title>
          <Text size="sm" c="dimmed" mt={4}>
            Run predictions with your trained model
          </Text>
        </div>

        <Grid gap="xl">
          {/* ── Left: Input & Results ── */}
          <Grid.Col span={{ base: 12, md: 8 }}>
            <Stack gap="md">
              {/* Input */}
              <Card withBorder padding="lg" radius="md">
                <Group justify="space-between" mb="md">
                  <Title order={5}>{isFileTask ? (isAudio ? 'Test Audio' : 'Test Image') : 'Test Input'}</Title>
                  {hasInput && (
                    <Tooltip label="Clear">
                      <ActionIcon variant="subtle" color="gray" onClick={handleClear}>
                        <ArrowCounterClockwiseIcon size={18} />
                      </ActionIcon>
                    </Tooltip>
                  )}
                </Group>

                {isFileTask && !inputFileUrl && (
                  <Dropzone
                    onDrop={handleDrop}
                    accept={descriptor.itemSpec.accept}
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
                            filter: isInferring ? 'brightness(0.6)' : 'none',
                          }}
                        />
                      )}
                      {isInferring && (
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

                {isTextTask && (
                  <Textarea
                    placeholder="Paste or type text to classify"
                    autosize
                    minRows={4}
                    value={textValue}
                    onChange={(e) => {
                      setTextValue(e.currentTarget.value)
                      setDetections(null)
                    }}
                  />
                )}

                {isRecordTask && (
                  <JsonInput
                    placeholder='{"age": 34, "income": 52000}'
                    description="One JSON object with a value for each feature column"
                    autosize
                    minRows={4}
                    formatOnBlur
                    value={recordJson}
                    onChange={(v) => {
                      setRecordJson(v)
                      setDetections(null)
                    }}
                  />
                )}

                {hasInput && (
                  <Group mt="md" gap="sm">
                    <Button
                      onClick={handleInfer}
                      loading={isInferring}
                      disabled={!selectedRunId}
                      leftSection={<LightningIcon size={18} weight="fill" />}
                      variant="gradient"
                      gradient={{ from: 'primary', to: 'secondary' }}
                    >
                      Run Inference
                    </Button>
                    <Button variant="subtle" color="gray" onClick={handleClear}>
                      Clear
                    </Button>
                  </Group>
                )}
              </Card>

              {/* Results */}
              <Transition mounted={!!filteredDetections} transition="slide-up" duration={300}>
                {(styles) => (
                  <Card withBorder padding="lg" radius="md" style={styles}>
                    <Group justify="space-between" mb="md">
                      <Group gap="sm">
                        <ThemeIcon variant="light" color="teal" size="sm">
                          <CrosshairIcon size={14} />
                        </ThemeIcon>
                        <Title order={5}>Results</Title>
                      </Group>
                      <Badge variant="light" color="teal" size="lg">
                        {filteredDetections?.length ?? 0} result(s)
                      </Badge>
                    </Group>

                    {filteredDetections && filteredDetections.length > 0 ? (
                      <Stack gap="xs">
                        {filteredDetections.map((d) => (
                          <Paper key={d.className} p="sm" radius="sm" withBorder>
                            <Group justify="space-between">
                              <Badge variant="filled" color="primary" size="sm">
                                {d.className}
                              </Badge>
                              <Group gap="xs">
                                <Progress
                                  value={d.confidence * 100}
                                  color={d.confidence > 0.8 ? 'teal' : d.confidence > 0.5 ? 'yellow' : 'red'}
                                  size="sm"
                                  w={80}
                                />
                                <Text size="xs" fw={600} w={45} ta="right">
                                  {(d.confidence * 100).toFixed(1)}%
                                </Text>
                              </Group>
                            </Group>
                          </Paper>
                        ))}

                        <Divider my="xs" />
                        <Code block style={{ maxHeight: 200, overflow: 'auto' }}>
                          {JSON.stringify(detections, null, 2)}
                        </Code>
                      </Stack>
                    ) : (
                      <Text size="sm" c="dimmed" ta="center" py="md">
                        No results above the confidence threshold
                      </Text>
                    )}
                  </Card>
                )}
              </Transition>
            </Stack>
          </Grid.Col>

          {/* ── Right: Controls ── */}
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Stack gap="md">
              {/* Model Selection */}
              <Card withBorder padding="lg" radius="md">
                <Title order={5} mb="md">
                  Trained Model
                </Title>
                <Select
                  label="Select a completed training run"
                  placeholder="Choose model..."
                  data={completedRuns.map((r) => ({
                    value: r.id,
                    label: r.name,
                  }))}
                  value={selectedRunId ?? null}
                  onChange={setSelectedRunId}
                  searchable
                />
                {completedRuns.length === 0 && (
                  <Text size="xs" c="dimmed" mt="sm">
                    No completed training runs available. Train a model first.
                  </Text>
                )}
              </Card>

              {/* Confidence Threshold */}
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

              <Alert color="gray" variant="light">
                Exporting a deployable model is handled on the{' '}
                <Link
                  to="/project/$projectId/models"
                  params={{ projectId }}
                  style={{ color: 'var(--mantine-color-primary-4)', textDecoration: 'underline' }}
                >
                  Models page
                </Link>
                .
              </Alert>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>
    </Box>
  )
}

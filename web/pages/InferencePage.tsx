import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Divider,
  Grid,
  Group,
  Loader,
  Overlay,
  Paper,
  Progress,
  Select,
  Slider,
  Stack,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
  Transition,
} from '@mantine/core'
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import {
  ArrowCounterClockwiseIcon,
  CheckCircleIcon,
  CloudArrowUpIcon,
  CrosshairIcon,
  DownloadSimpleIcon,
  ExportIcon,
  LightningIcon,
  WarningCircleIcon,
  XCircleIcon,
} from '@phosphor-icons/react'
import { api } from '@public/lib/api'
import { useTrainingRuns } from '@public/queries/training'
import { useState } from 'react'
import { useParams } from 'wouter'

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

/** Raw inference result: { className: confidence } */
type InferenceResult = Record<string, number>

interface DetectionEntry {
  className: string
  confidence: number
}

const EXPORT_FORMATS = [
  { value: 'onnx', label: 'ONNX (.onnx)' },
  { value: 'torchscript', label: 'TorchScript (.pt)' },
]

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function InferencePage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id

  // Image state
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [detections, setDetections] = useState<InferenceResult | null>(null)

  // Confidence threshold
  const [threshold, setThreshold] = useState(0.5)

  // Selected model (training run)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)

  // Status
  const [isInferring, setIsInferring] = useState(false)

  // Export
  const [exportFormat, setExportFormat] = useState('onnx')

  // Fetch succeeded training runs
  const { data: runsData } = useTrainingRuns(projectId)
  const completedRuns = (runsData?.runs ?? []).filter((r) => r.status === 'succeeded')

  /* ── Handlers ──────────────────────────────────────────────────── */
  const handleDrop = (files: File[]) => {
    if (files.length === 0) return
    const file = files[0]
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
    setDetections(null)
  }

  const handleInfer = async () => {
    if (!imageFile || !selectedRunId) {
      notifications.show({ title: 'Not Ready', message: 'Please select a model and upload an image.', color: 'orange' })
      return
    }
    setIsInferring(true)
    setDetections(null)

    try {
      const response = await api.inference({ runId: selectedRunId }).post({
        image: imageFile,
        threshold,
      })

      if (response.error) {
        notifications.show({
          title: 'Inference Failed',
          message: typeof response.error.value === 'string' ? response.error.value : 'Unknown error',
          color: 'red',
          icon: <XCircleIcon size={18} />,
        })
      } else if (response.data) {
        const data = response.data as any
        if (data.status === 'success' && data.results) {
          setDetections(data.results)
          notifications.show({
            title: 'Inference Complete',
            message: `${Object.keys(data.results).length} class(es) predicted`,
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

  const handleDownload = (format: string) => {
    if (!selectedRunId) return
    const a = document.createElement('a')
    a.href = `/api/runs/${selectedRunId}/download/${format}`
    a.download = `model.${format}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleClear = () => {
    setImageUrl(null)
    setImageFile(null)
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
          <Title order={2}>Inference & Export</Title>
          <Text size="sm" c="dimmed" mt={4}>
            Run predictions with your trained model and export weights
          </Text>
        </div>

        <Grid gap="xl">
          {/* ── Left: Image & Results ── */}
          <Grid.Col span={{ base: 12, md: 8 }}>
            <Stack gap="md">
              {/* Upload / Preview */}
              <Card withBorder padding="lg" radius="md">
                <Group justify="space-between" mb="md">
                  <Title order={5}>Test Image</Title>
                  {imageUrl && (
                    <Tooltip label="Clear image">
                      <ActionIcon variant="subtle" color="gray" onClick={handleClear}>
                        <ArrowCounterClockwiseIcon size={18} />
                      </ActionIcon>
                    </Tooltip>
                  )}
                </Group>

                {!imageUrl ? (
                  <Dropzone
                    onDrop={handleDrop}
                    accept={IMAGE_MIME_TYPE}
                    radius="md"
                    p="xl"
                    style={{
                      borderStyle: 'dashed',
                      borderWidth: 2,
                      transition: 'all 150ms ease',
                    }}
                  >
                    <Stack align="center" gap="sm">
                      <ThemeIcon
                        size={56}
                        variant="light"
                        color="primary"
                        radius="xl"
                        style={{ transition: 'transform 200ms ease' }}
                      >
                        <CloudArrowUpIcon size={30} />
                      </ThemeIcon>
                      <Text size="sm" fw={500}>
                        Drop an image here or click to browse
                      </Text>
                      <Text size="xs" c="dimmed">
                        Supports JPEG, PNG, WebP, GIF
                      </Text>
                    </Stack>
                  </Dropzone>
                ) : (
                  <Box pos="relative">
                    <Paper radius="md" style={{ overflow: 'hidden', position: 'relative' }}>
                      <img
                        src={imageUrl}
                        alt="Test"
                        style={{
                          width: '100%',
                          display: 'block',
                          transition: 'filter 300ms ease',
                          filter: isInferring ? 'brightness(0.6)' : 'none',
                        }}
                      />
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
                  </Box>
                )}
              </Card>

              {/* Detection Results */}
              <Transition mounted={!!filteredDetections} transition="slide-up" duration={300}>
                {(styles) => (
                  <Card withBorder padding="lg" radius="md" style={styles}>
                    <Group justify="space-between" mb="md">
                      <Group gap="sm">
                        <ThemeIcon variant="light" color="teal" size="sm">
                          <CrosshairIcon size={14} />
                        </ThemeIcon>
                        <Title order={5}>Detection Results</Title>
                      </Group>
                      <Badge variant="light" color="teal" size="lg">
                        {filteredDetections?.length ?? 0} class(es)
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
                        No detections above the confidence threshold
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
                  value={selectedRunId}
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
                  Filter out detections below this confidence score
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

              {/* Export Model */}
              <Card withBorder padding="lg" radius="md">
                <Group gap="sm" mb="md">
                  <ThemeIcon variant="light" color="secondary" size="sm">
                    <ExportIcon size={14} />
                  </ThemeIcon>
                  <Title order={5}>Export Model</Title>
                </Group>
                <Text size="sm" c="dimmed" mb="md">
                  Download trained weights for deployment
                </Text>
                <Select
                  label="Export Format"
                  data={EXPORT_FORMATS}
                  value={exportFormat}
                  onChange={(v) => setExportFormat(v!)}
                  mb="md"
                />
                <Button
                  fullWidth
                  variant="gradient"
                  gradient={{ from: 'secondary.5', to: 'primary.5' }}
                  leftSection={<DownloadSimpleIcon size={18} />}
                  onClick={() => handleDownload(exportFormat)}
                  disabled={!selectedRunId}
                >
                  Download {exportFormat.toUpperCase()}
                </Button>
              </Card>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>
    </Box>
  )
}

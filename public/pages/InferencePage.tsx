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
import { useCallback, useEffect, useRef, useState } from 'react'
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

type InferenceStatus = 'idle' | 'connecting' | 'connected' | 'inferring' | 'exporting'

interface ExportJob {
  jobId: string
  format: string
  status: 'pending' | 'success' | 'failed'
  error?: string
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

  // WebSocket
  const wsRef = useRef<WebSocket | null>(null)
  const [wsStatus, setWsStatus] = useState<InferenceStatus>('idle')
  const [wsConnected, setWsConnected] = useState(false)

  // Export
  const [exportFormat, setExportFormat] = useState('onnx')
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([])

  // Reconnect
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const mountedRef = useRef(true)

  /* ── WebSocket connection ──────────────────────────────────────── */
  const connectWs = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setWsStatus('connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/inference/${projectId}/ws`)

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return }
      setWsConnected(true)
      setWsStatus('connected')
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)

        switch (msg.type) {
          case 'connected':
            break

          case 'inference_queued':
            // Job dispatched, wait for result
            break

          case 'inference_result':
            setWsStatus('connected')
            if (msg.status === 'success' && msg.result) {
              // Result is a dict: { className: confidence }
              const resultDict: InferenceResult = msg.result
              setDetections(resultDict)
              notifications.show({
                title: 'Inference Complete',
                message: `${Object.keys(resultDict).length} class(es) predicted`,
                color: 'teal',
                icon: <CheckCircleIcon size={18} />,
              })
            } else {
              notifications.show({
                title: 'Inference Failed',
                message: msg.error ?? 'Unknown error',
                color: 'red',
                icon: <XCircleIcon size={18} />,
              })
            }
            break

          case 'export_queued':
            setExportJobs((prev) => [
              ...prev,
              { jobId: msg.jobId, format: msg.format, status: 'pending' },
            ])
            notifications.show({
              title: 'Export Started',
              message: `Exporting model to ${msg.format.toUpperCase()}...`,
              color: 'blue',
              icon: <ExportIcon size={18} />,
            })
            break

          case 'export_result':
            setWsStatus('connected')
            setExportJobs((prev) =>
              prev.map((j) =>
                j.jobId === msg.jobId
                  ? { ...j, status: msg.status, error: msg.error }
                  : j,
              ),
            )
            if (msg.status === 'success') {
              notifications.show({
                title: 'Export Complete',
                message: 'Model exported successfully! Click download below.',
                color: 'teal',
                icon: <CheckCircleIcon size={18} />,
              })
            } else {
              notifications.show({
                title: 'Export Failed',
                message: msg.error ?? 'Unknown error',
                color: 'red',
                icon: <XCircleIcon size={18} />,
              })
            }
            break

          case 'error':
            setWsStatus('connected')
            notifications.show({
              title: 'Error',
              message: msg.message,
              color: 'red',
              icon: <WarningCircleIcon size={18} />,
            })
            break
        }
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      setWsConnected(false)
      setWsStatus('idle')
      // Auto-reconnect after 3s
      if (mountedRef.current) {
        reconnectTimeoutRef.current = setTimeout(connectWs, 3000)
      }
    }

    ws.onerror = () => {
      ws.close()
    }

    wsRef.current = ws
  }, [projectId])

  useEffect(() => {
    mountedRef.current = true
    connectWs()
    return () => {
      mountedRef.current = false
      clearTimeout(reconnectTimeoutRef.current)
      wsRef.current?.close()
    }
  }, [connectWs])

  /* ── Handlers ──────────────────────────────────────────────────── */
  const handleDrop = (files: File[]) => {
    if (files.length === 0) return
    const file = files[0]
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
    setDetections(null)
  }

  const handleInfer = async () => {
    if (!imageFile || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      notifications.show({ title: 'Not Ready', message: 'WebSocket is not connected.', color: 'orange' })
      return
    }
    setWsStatus('inferring')
    setDetections(null)

    // Read file as base64
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result as string
      wsRef.current?.send(
        JSON.stringify({
          type: 'inference',
          image: base64,
          imageName: imageFile.name,
          threshold,
        }),
      )
    }
    reader.readAsDataURL(imageFile)
  }

  const handleExport = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      notifications.show({ title: 'Not Ready', message: 'WebSocket is not connected.', color: 'orange' })
      return
    }
    setWsStatus('exporting')
    wsRef.current.send(
      JSON.stringify({
        type: 'export',
        format: exportFormat,
      }),
    )
  }

  const handleDownload = (format: string) => {
    const a = document.createElement('a')
    a.href = `/api/inference/${projectId}/download/${format}`
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

  const isInferring = wsStatus === 'inferring'
  const isExporting = wsStatus === 'exporting'

  /* ── Render ────────────────────────────────────────────────────── */
  return (
    <Box>
      <Stack gap="xl">
        {/* Page header */}
        <Group justify="space-between" align="flex-end">
          <div>
            <Title order={2}>Inference & Export</Title>
            <Text size="sm" c="dimmed" mt={4}>
              Run predictions with your trained model and export weights
            </Text>
          </div>
          <Badge
            size="lg"
            variant="dot"
            color={wsConnected ? 'teal' : 'red'}
            style={{ textTransform: 'none' }}
          >
            {wsConnected ? 'WebSocket Connected' : 'Disconnected'}
          </Badge>
        </Group>

        {/* Warning if not connected */}
        <Transition mounted={!wsConnected} transition="slide-down" duration={200}>
          {(styles) => (
            <Card withBorder p="lg" radius="md" bg="dark.7" style={styles}>
              <Group gap="sm">
                <ThemeIcon variant="light" color="yellow">
                  <WarningCircleIcon size={18} />
                </ThemeIcon>
                <div>
                  <Text size="sm" fw={500}>
                    WebSocket disconnected
                  </Text>
                  <Text size="xs" c="dimmed">
                    Attempting to reconnect automatically...
                  </Text>
                </div>
              </Group>
            </Card>
          )}
        </Transition>

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
                    <Paper
                      radius="md"
                      style={{ overflow: 'hidden', position: 'relative' }}
                    >
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
                        <Overlay
                          center
                          backgroundOpacity={0.4}
                          blur={2}
                          radius="md"
                        >
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
                        disabled={!wsConnected}
                        leftSection={<LightningIcon size={18} weight="fill" />}
                        variant="gradient"
                        gradient={{ from: 'primary', to: 'secondary' }}
                      >
                        Run Inference
                      </Button>
                      <Button
                        variant="subtle"
                        color="gray"
                        onClick={handleClear}
                      >
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
                              <Group gap="sm">
                                <Badge variant="filled" color="primary" size="sm">
                                  {d.className}
                                </Badge>
                              </Group>
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
                  Export trained weights for deployment on mobile or edge devices
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
                  leftSection={<ExportIcon size={18} />}
                  onClick={handleExport}
                  loading={isExporting}
                  disabled={!wsConnected}
                >
                  Start Export
                </Button>

                {/* Export History */}
                {exportJobs.length > 0 && (
                  <>
                    <Divider my="md" label="Export History" labelPosition="center" />
                    <Stack gap="xs">
                      {exportJobs.map((job) => (
                        <Paper key={job.jobId} p="xs" radius="sm" withBorder>
                          <Group justify="space-between">
                            <Group gap="xs">
                              <Badge
                                size="xs"
                                variant="light"
                                color={
                                  job.status === 'success'
                                    ? 'teal'
                                    : job.status === 'failed'
                                      ? 'red'
                                      : 'yellow'
                                }
                              >
                                {job.status === 'pending' && <Loader size={8} mr={4} />}
                                {job.status}
                              </Badge>
                              <Text size="xs" fw={500}>
                                {job.format.toUpperCase()}
                              </Text>
                            </Group>
                            {job.status === 'success' && (
                              <Tooltip label={`Download ${job.format.toUpperCase()}`}>
                                <ActionIcon
                                  variant="light"
                                  color="teal"
                                  size="sm"
                                  onClick={() => handleDownload(job.format)}
                                >
                                  <DownloadSimpleIcon size={14} />
                                </ActionIcon>
                              </Tooltip>
                            )}
                            {job.status === 'failed' && (
                              <Tooltip label={job.error ?? 'Export failed'}>
                                <ThemeIcon variant="light" color="red" size="sm">
                                  <XCircleIcon size={14} />
                                </ThemeIcon>
                              </Tooltip>
                            )}
                          </Group>
                        </Paper>
                      ))}
                    </Stack>
                  </>
                )}
              </Card>

              {/* Connection Info */}
              <Card withBorder padding="lg" radius="md" bg="dark.8">
                <Title order={6} mb="xs" c="dimmed">
                  Connection Info
                </Title>
                <Stack gap={4}>
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">
                      Protocol
                    </Text>
                    <Badge size="xs" variant="outline" color="gray">
                      WebSocket
                    </Badge>
                  </Group>
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">
                      Status
                    </Text>
                    <Badge
                      size="xs"
                      variant="dot"
                      color={wsConnected ? 'teal' : 'red'}
                    >
                      {wsStatus}
                    </Badge>
                  </Group>
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">
                      Delivery
                    </Text>
                    <Badge size="xs" variant="outline" color="gray">
                      Webhook → WS Push
                    </Badge>
                  </Group>
                </Stack>
              </Card>
            </Stack>
          </Grid.Col>
        </Grid>
      </Stack>
    </Box>
  )
}

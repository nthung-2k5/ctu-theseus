/**
 * "Import a model", in three steps, full screen like the new-project modal:
 *
 *   1. Source   an uploaded .zip of the model folder, or a Hugging Face Hub id
 *   2. Check    what the model is: read from its config.json (the Hub's, or the one inside the zip, read in the
 *               browser before anything is uploaded), shown as a result card. The kind is filled in when
 *               config.json says; when it does not that is only a warning and the person picks it.
 *   3. Details  name and description, then import
 *
 * There is no task to pick: a model is offered for every task its kind serves, which the backend reports
 * (GET /api/models/kinds) and this form only displays.
 *
 * For an upload the model is created first and the file follows: if the upload then fails the model
 * still exists, waiting for its file, and the table offers "Upload file" to try again.
 */

import {
  Alert,
  Anchor,
  Badge,
  Button,
  Group,
  Modal,
  Paper,
  Progress,
  Select,
  SimpleGrid,
  Stack,
  Stepper,
  Text,
  Textarea,
  TextInput,
  ThemeIcon,
  UnstyledButton,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  CloudArrowDownIcon,
  FileZipIcon,
  WarningCircleIcon,
} from '@phosphor-icons/react'
import { apiErrorMessage } from '@public/lib/api/client'
import type { CustomModelKindOut, InspectModelResponse } from '@public/lib/api/generated/models'
import { getListCustomModelKindsQueryOptions, inspectModel } from '@public/lib/api/generated/models/models'
import { formatBytes, IMPORTED_MODELS_KEY, type ModelsApi, taskLabel } from '@public/lib/importedModels'
import { readZipConfig } from '@public/lib/modelZip'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { ModelFileDropzone } from './ModelFileDropzone'
import { useModelUpload } from './useModelUpload'

type Source = 'upload' | 'hub'

const kindKey = (k: CustomModelKindOut) => `${k.backend}:${k.id}`

const NO_CONFIG_WARNING =
  'Could not read a config.json from the start of the zip, so the kind of model cannot be told from it. ' +
  'Choose the kind yourself; the server checks the files after the upload.'

const EMPTY_INSPECTION: InspectModelResponse = {
  sourceRef: null,
  revision: null,
  sizeBytes: null,
  modelType: null,
  architectures: [],
  suggestedBackend: null,
  suggestedKind: null,
  errors: [],
  warnings: [],
}

/** A default name for the model: the repository's own name, or the zip's file name. */
function defaultName(source: Source, hubId: string, file: File | null): string {
  if (source === 'hub') return hubId.trim().split('/').pop() ?? ''
  return (file?.name ?? '').replace(/\.zip$/i, '')
}

function selectedStyle(selected: boolean) {
  return {
    borderColor: selected ? 'var(--mantine-color-primary-filled)' : undefined,
    backgroundColor: selected ? 'var(--mantine-color-primary-light)' : undefined,
    transition: 'all 150ms ease',
  }
}

export function ImportedModelForm({
  api,
  opened,
  onClose,
  title,
}: {
  api: ModelsApi
  opened: boolean
  onClose: () => void
  title: string
}) {
  const queryClient = useQueryClient()
  const { data } = useQuery(getListCustomModelKindsQueryOptions())
  const kinds = data?.kinds ?? []
  const { upload, progress } = useModelUpload(api)

  const [step, setStep] = useState(0)
  const [source, setSource] = useState<Source>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [hubId, setHubId] = useState('')
  const [inspection, setInspection] = useState<InspectModelResponse | null>(null)
  const [kindValue, setKindValue] = useState('')
  const [changingKind, setChangingKind] = useState(false)
  // A Hub id that is really a timm model name: nothing on the Hub to inspect, so the check's errors are moot.
  const [byName, setByName] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const busy = progress !== null
  const fromSource = kinds.filter((k) => k.sourceKinds.includes(source))
  const kind = kinds.find((k) => kindKey(k) === kindValue)
  const detected = inspection?.suggestedBackend ? `${inspection.suggestedBackend}:${inspection.suggestedKind}` : null
  const byNameKind = fromSource.find((k) => !k.needsFiles && !k.unavailableReason)
  const blocked = !!inspection?.errors.length && !byName
  const canContinue = !!kind && !kind.unavailableReason && !blocked

  const reset = () => {
    setStep(0)
    setSource('upload')
    setFile(null)
    setHubId('')
    setInspection(null)
    setKindValue('')
    setChangingKind(false)
    setByName(false)
    setName('')
    setDescription('')
  }

  const close = () => {
    reset()
    onClose()
  }

  const chooseSource = (next: Source) => {
    if (next === source) return
    setSource(next)
    setInspection(null)
    setKindValue('')
  }

  const check = useMutation({
    mutationFn: async (): Promise<InspectModelResponse> => {
      if (source === 'hub') return inspectModel({ sourceRef: hubId.trim() })
      const config = file ? await readZipConfig(file) : null
      if (config) return inspectModel({ config })
      return { ...EMPTY_INSPECTION, warnings: [NO_CONFIG_WARNING] }
    },
    onSuccess: (result) => {
      setInspection(result)
      setKindValue(result.suggestedBackend ? `${result.suggestedBackend}:${result.suggestedKind}` : '')
      setChangingKind(false)
      setByName(false)
      setStep(1)
    },
    onError: (error) =>
      notifications.show({ title: 'Could not check the model', message: apiErrorMessage(error), color: 'red' }),
  })

  const create = useMutation({
    mutationFn: async () => {
      if (!kind) throw new Error('Choose what kind of model this is')
      const model = await api.create({
        name: name.trim(),
        description: description.trim(),
        backend: kind.backend,
        kind: kind.id,
        sourceKind: source,
        sourceRef: source === 'hub' ? hubId.trim() : null,
      })
      let uploadError: unknown = null
      if (source === 'upload' && file) {
        try {
          await upload(model.id, file)
        } catch (error) {
          uploadError = error
        }
      }
      return { model, uploadError }
    },
    onSuccess: ({ model, uploadError }) => {
      queryClient.invalidateQueries({ queryKey: IMPORTED_MODELS_KEY })
      if (uploadError) {
        notifications.show({
          title: 'Model imported, upload failed',
          message: `${apiErrorMessage(uploadError)} Use “Upload file” on “${model.name}” to try again.`,
          color: 'orange',
        })
      } else {
        notifications.show({
          title: 'Model imported',
          message: 'It will be checked now. You can fine-tune it once it shows as Ready.',
          color: 'teal',
        })
      }
      close()
    },
    onError: (error) =>
      notifications.show({ title: 'Could not import the model', message: apiErrorMessage(error), color: 'red' }),
  })

  const sourceReady = source === 'upload' ? !!file : !!hubId.trim()
  const startCheck = () => {
    if (sourceReady && !check.isPending) check.mutate()
  }

  const toDetails = () => {
    if (!name.trim()) setName(defaultName(source, hubId, file))
    setStep(2)
  }

  return (
    <Modal
      opened={opened}
      onClose={busy ? () => undefined : close}
      title={title}
      fullScreen
      closeOnClickOutside={!busy}
      closeOnEscape={!busy}
      styles={{ title: { fontSize: 'var(--mantine-font-size-xl)', fontWeight: 700 } }}
    >
      <div className="mx-auto w-full max-w-3xl pb-8">
        <Stepper active={step} size="sm" mb="xl" allowNextStepsSelect={false}>
          <Stepper.Step label="Source" description="Where the model is" />
          <Stepper.Step label="Check" description="What it is" />
          <Stepper.Step label="Details" description="Name it" />
        </Stepper>

        {step === 0 && (
          <Stack gap="lg">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md" role="radiogroup" aria-label="Source">
              <SourceCard
                selected={source === 'upload'}
                icon={<FileZipIcon size={26} />}
                title="Upload a .zip"
                description="The model folder zipped: config.json, tokenizer files and .safetensors or .pth weights."
                onClick={() => chooseSource('upload')}
              />
              <SourceCard
                selected={source === 'hub'}
                icon={<CloudArrowDownIcon size={26} />}
                title="Hugging Face Hub"
                description="A public repository. Its current version is pinned when it is imported."
                onClick={() => chooseSource('hub')}
              />
            </SimpleGrid>

            {source === 'upload' ? (
              <ModelFileDropzone file={file} onChange={setFile} disabled={check.isPending} />
            ) : (
              <TextInput
                label="Model id"
                placeholder="org/model-name"
                autoFocus
                value={hubId}
                onChange={(e) => setHubId(e.currentTarget.value)}
                onKeyDown={(e) => e.key === 'Enter' && startCheck()}
              />
            )}

            <Group justify="flex-end">
              <Button variant="subtle" onClick={close}>
                Cancel
              </Button>
              <Button
                rightSection={<ArrowRightIcon size={16} />}
                loading={check.isPending}
                disabled={!sourceReady}
                onClick={startCheck}
              >
                Check model
              </Button>
            </Group>
          </Stack>
        )}

        {step === 1 && inspection && (
          <Stack gap="lg">
            <Paper withBorder radius="md" p="lg">
              <Stack gap="md">
                <Group gap="md" wrap="nowrap" align="flex-start">
                  <ThemeIcon size={44} radius="md" variant="light" color={blocked ? 'red' : 'primary'}>
                    {source === 'upload' ? <FileZipIcon size={24} /> : <CloudArrowDownIcon size={24} />}
                  </ThemeIcon>
                  <div className="min-w-0 flex-1">
                    <Text fw={700} truncate>
                      {source === 'upload' ? file?.name : hubId.trim()}
                    </Text>
                    <Text size="sm" c="dimmed">
                      {describeInspection(inspection, source === 'upload' ? file?.size : undefined)}
                    </Text>
                  </div>
                  {blocked ? (
                    <Badge color="red" variant="light">
                      Cannot import
                    </Badge>
                  ) : (
                    <Badge color="teal" variant="light" leftSection={<CheckCircleIcon size={12} />}>
                      Looks good
                    </Badge>
                  )}
                </Group>

                {!byName &&
                  inspection.errors.map((message) => (
                    <Alert key={message} color="red" icon={<WarningCircleIcon size={16} />} p="xs">
                      {message}
                    </Alert>
                  ))}
                {inspection.warnings.map((message) => (
                  <Alert key={message} color="orange" icon={<WarningCircleIcon size={16} />} p="xs">
                    {message}
                  </Alert>
                ))}
                {blocked && source === 'hub' && byNameKind && (
                  <Text size="sm" c="dimmed">
                    Is this the name of a model in a library rather than a Hub repository?{' '}
                    <Anchor
                      component="button"
                      type="button"
                      size="sm"
                      onClick={() => {
                        setByName(true)
                        setKindValue(kindKey(byNameKind))
                        setChangingKind(false)
                      }}
                    >
                      Import it as “{byNameKind.label}”
                    </Anchor>
                  </Text>
                )}
              </Stack>
            </Paper>

            {(!blocked || byName) && (
              <Paper withBorder radius="md" p="lg">
                <Stack gap="sm">
                  {kind && !changingKind ? (
                    <Group justify="space-between" wrap="nowrap">
                      <div>
                        <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                          {kindValue === detected ? 'Detected' : 'Kind of model'}
                        </Text>
                        <Group gap={6}>
                          <Text fw={700}>{kind.label}</Text>
                          {kind.status === 'experimental' && (
                            <Badge size="xs" color="orange" variant="light">
                              experimental
                            </Badge>
                          )}
                        </Group>
                      </div>
                      {!byName && (
                        <Anchor component="button" type="button" size="sm" onClick={() => setChangingKind(true)}>
                          Change
                        </Anchor>
                      )}
                    </Group>
                  ) : (
                    <Select
                      label="Kind of model"
                      placeholder={kinds.length ? 'Choose…' : 'Loading…'}
                      data={fromSource.map((k) => ({
                        value: kindKey(k),
                        label: k.label,
                        disabled: !!k.unavailableReason,
                      }))}
                      value={kindValue || null}
                      onChange={(value) => {
                        setKindValue(value ?? '')
                        if (value) setChangingKind(false)
                      }}
                      allowDeselect={false}
                      comboboxProps={{ withinPortal: true }}
                    />
                  )}

                  {kind && (
                    <>
                      <Text size="sm" c="dimmed">
                        {kind.description}
                      </Text>
                      {kind.unavailableReason && (
                        <Alert color="orange" icon={<WarningCircleIcon size={16} />} p="xs">
                          {kind.unavailableReason}
                        </Alert>
                      )}
                      <div>
                        <Text size="xs" c="dimmed" mb={4}>
                          Available for these tasks
                        </Text>
                        <Group gap={6}>
                          {kind.tasks.map((t) => (
                            <Badge key={t} variant="outline" color="gray">
                              {taskLabel(t)}
                            </Badge>
                          ))}
                        </Group>
                      </div>
                    </>
                  )}
                </Stack>
              </Paper>
            )}

            <Group justify="space-between">
              <Button variant="subtle" leftSection={<ArrowLeftIcon size={16} />} onClick={() => setStep(0)}>
                Back
              </Button>
              <Button rightSection={<ArrowRightIcon size={16} />} disabled={!canContinue} onClick={toDetails}>
                Next
              </Button>
            </Group>
          </Stack>
        )}

        {step === 2 && (
          <Stack gap="lg">
            <Paper withBorder radius="md" p="lg">
              <Stack gap="md">
                <TextInput
                  label="Name"
                  placeholder="e.g. Legal-BERT"
                  data-autofocus
                  value={name}
                  onChange={(e) => setName(e.currentTarget.value)}
                  error={name.trim() ? null : 'Give the imported model a name'}
                />
                <Textarea
                  label="Description"
                  placeholder="Optional"
                  autosize
                  minRows={2}
                  maxRows={6}
                  value={description}
                  onChange={(e) => setDescription(e.currentTarget.value)}
                />
              </Stack>
            </Paper>

            <Paper withBorder radius="md" p="md">
              <Stack gap={4}>
                <SummaryRow label="Source" value={source === 'upload' ? (file?.name ?? '') : hubId.trim()} />
                <SummaryRow label="Kind" value={kind?.label ?? ''} />
                <SummaryRow label="Available for" value={(kind?.tasks ?? []).map(taskLabel).join(', ')} />
              </Stack>
            </Paper>

            {progress !== null && (
              <Stack gap={2}>
                <Progress value={progress * 100} animated />
                <Text size="xs" c="dimmed">
                  Uploading… {Math.round(progress * 100)}%
                </Text>
              </Stack>
            )}

            <Group justify="space-between">
              <Button
                variant="subtle"
                leftSection={<ArrowLeftIcon size={16} />}
                onClick={() => setStep(1)}
                disabled={busy}
              >
                Back
              </Button>
              <Button
                loading={create.isPending}
                disabled={!name.trim() || !canContinue}
                onClick={() => create.mutate()}
              >
                {source === 'upload' ? 'Import and upload' : 'Import model'}
              </Button>
            </Group>
          </Stack>
        )}
      </div>
    </Modal>
  )
}

function SourceCard({
  selected,
  icon,
  title,
  description,
  onClick,
}: {
  selected: boolean
  icon: React.ReactNode
  title: string
  description: string
  onClick: () => void
}) {
  return (
    <UnstyledButton role="radio" aria-checked={selected} onClick={onClick}>
      <Paper withBorder p="md" radius="md" h="100%" style={selectedStyle(selected)}>
        <Group gap="md" wrap="nowrap" align="flex-start">
          <ThemeIcon size={44} radius="md" variant={selected ? 'filled' : 'light'}>
            {icon}
          </ThemeIcon>
          <div>
            <Text fw={700}>{title}</Text>
            <Text size="sm" c="dimmed">
              {description}
            </Text>
          </div>
        </Group>
      </Paper>
    </UnstyledButton>
  )
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <Group gap="md" wrap="nowrap" align="flex-start">
      <Text size="sm" c="dimmed" w={110} style={{ flexShrink: 0 }}>
        {label}
      </Text>
      <Text size="sm">{value}</Text>
    </Group>
  )
}

/** The facts the check found, on one line: family, architecture, size, pinned revision. */
function describeInspection(result: InspectModelResponse, fileBytes?: number): string {
  const size = result.sizeBytes ?? fileBytes
  return (
    [
      result.modelType,
      result.architectures.join(', '),
      size != null && formatBytes(size),
      result.revision && `revision ${result.revision.slice(0, 7)}`,
    ]
      .filter(Boolean)
      .join(' · ') || 'No details in config.json'
  )
}

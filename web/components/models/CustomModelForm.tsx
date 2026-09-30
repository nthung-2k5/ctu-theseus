/**
 * "Add a model": what it is (a kind the backend understands), where its weights come from (a Hugging Face
 * Hub id, or an uploaded file) and which tasks it is offered for. The kinds and their tasks come from the
 * backend (GET /api/models/kinds), so this form never hard-codes what a backend supports.
 *
 * For an upload the model is created first and the file follows: if the upload then fails the model
 * still exists, waiting for its file, and the table offers "Upload file" to try again.
 */

import {
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  MultiSelect,
  Progress,
  SegmentedControl,
  Select,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { WarningCircleIcon } from '@phosphor-icons/react'
import { apiErrorMessage } from '@public/lib/api/client'
import type { CustomModelKindOut } from '@public/lib/api/generated/models'
import { getListCustomModelKindsQueryOptions } from '@public/lib/api/generated/models/models'
import { CUSTOM_MODELS_KEY, type ModelsApi, taskLabel } from '@public/lib/customModels'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { ModelFileDropzone } from './ModelFileDropzone'
import { useModelUpload } from './useModelUpload'

const kindKey = (k: CustomModelKindOut) => `${k.backend}:${k.id}`

interface FormValues {
  name: string
  description: string
  kind: string
  sourceKind: 'hub' | 'upload'
  sourceRef: string
  tasks: string[]
}

const INITIAL: FormValues = { name: '', description: '', kind: '', sourceKind: 'hub', sourceRef: '', tasks: [] }

export function CustomModelForm({
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
  const [file, setFile] = useState<File | null>(null)
  const { upload, progress } = useModelUpload(api)

  const form = useForm<FormValues>({
    initialValues: INITIAL,
    validate: {
      name: (v) => (v.trim() ? null : 'Give the model a name'),
      kind: (v) => (v ? null : 'Choose what kind of model this is'),
      tasks: (v) => (v.length > 0 ? null : 'Choose at least one task'),
      sourceRef: (v, values) => (values.sourceKind === 'hub' && !v.trim() ? 'Enter the model id' : null),
    },
  })

  const kind = kinds.find((k) => kindKey(k) === form.values.kind)
  const busy = progress !== null

  const close = () => {
    form.reset()
    setFile(null)
    onClose()
  }

  const chooseKind = (value: string | null) => {
    const next = kinds.find((k) => kindKey(k) === value)
    form.setValues({
      kind: value ?? '',
      // Keep the current source if this kind allows it; otherwise fall back to the first one it does.
      sourceKind:
        next && !next.sourceKinds.includes(form.values.sourceKind) ? next.sourceKinds[0] : form.values.sourceKind,
      // A sensible default: offered for every task the kind supports. The person can narrow it.
      tasks: next?.tasks ?? [],
    })
  }

  const create = useMutation({
    mutationFn: async (values: FormValues) => {
      if (!kind) throw new Error('Choose what kind of model this is')
      const model = await api.create({
        name: values.name.trim(),
        description: values.description.trim(),
        backend: kind.backend,
        kind: kind.id,
        sourceKind: values.sourceKind,
        sourceRef: values.sourceKind === 'hub' ? values.sourceRef.trim() : null,
        tasks: values.tasks,
      })
      let uploadError: unknown = null
      if (values.sourceKind === 'upload' && file) {
        try {
          await upload(model.id, file)
        } catch (error) {
          uploadError = error
        }
      }
      return { model, uploadError }
    },
    onSuccess: ({ model, uploadError }) => {
      queryClient.invalidateQueries({ queryKey: CUSTOM_MODELS_KEY })
      if (uploadError) {
        notifications.show({
          title: 'Model created, upload failed',
          message: `${apiErrorMessage(uploadError)} Use “Upload file” on “${model.name}” to try again.`,
          color: 'orange',
        })
      } else {
        notifications.show({
          title: 'Model added',
          message: 'It will be checked now. You can train on it once it shows as Ready.',
          color: 'teal',
        })
      }
      close()
    },
    onError: (error) =>
      notifications.show({ title: 'Could not add the model', message: apiErrorMessage(error), color: 'red' }),
  })

  const submit = form.onSubmit((values) => {
    if (values.sourceKind === 'upload' && !file) {
      form.setFieldError('sourceKind', 'Choose the file to upload')
      return
    }
    create.mutate(values)
  })

  const kindOptions = Object.values(
    kinds.reduce<Record<string, { group: string; items: { value: string; label: string; disabled: boolean }[] }>>(
      (acc, k) => {
        const group = acc[k.backend] ?? { group: k.backendLabel, items: [] }
        group.items.push({ value: kindKey(k), label: k.label, disabled: !!k.unavailableReason })
        acc[k.backend] = group
        return acc
      },
      {},
    ),
  )

  return (
    <Modal
      opened={opened}
      onClose={busy ? () => undefined : close}
      title={title}
      centered
      size="lg"
      closeOnClickOutside={!busy}
    >
      <form onSubmit={submit}>
        <Stack gap="sm">
          <TextInput label="Name" placeholder="e.g. Legal-BERT" data-autofocus {...form.getInputProps('name')} />
          <Textarea
            label="Description"
            placeholder="Optional"
            autosize
            minRows={1}
            maxRows={4}
            {...form.getInputProps('description')}
          />

          <Select
            label="Kind of model"
            placeholder={kinds.length ? 'Choose…' : 'Loading…'}
            data={kindOptions}
            value={form.values.kind || null}
            onChange={chooseKind}
            allowDeselect={false}
            error={form.errors.kind}
          />
          {kind && (
            <Stack gap={4}>
              <Group gap={6}>
                <Text size="xs" c="dimmed">
                  {kind.description}
                </Text>
                {kind.status === 'experimental' && (
                  <Badge size="xs" color="orange" variant="light">
                    experimental
                  </Badge>
                )}
              </Group>
              {kind.unavailableReason && (
                <Alert color="orange" icon={<WarningCircleIcon size={16} />} p="xs">
                  {kind.unavailableReason}
                </Alert>
              )}
            </Stack>
          )}

          {kind && kind.sourceKinds.length > 1 && (
            <SegmentedControl
              value={form.values.sourceKind}
              onChange={(v) => form.setFieldValue('sourceKind', v as 'hub' | 'upload')}
              disabled={busy}
              data={[
                { value: 'hub', label: 'Hugging Face Hub' },
                { value: 'upload', label: 'Upload files' },
              ]}
            />
          )}

          {kind && form.values.sourceKind === 'hub' && (
            <TextInput
              label="Model id"
              description="A public repository on the Hugging Face Hub. Its current version is pinned when it is checked."
              placeholder="org/model-name"
              {...form.getInputProps('sourceRef')}
            />
          )}
          {kind && form.values.sourceKind === 'upload' && (
            <Stack gap={4}>
              <ModelFileDropzone file={file} onChange={setFile} disabled={busy} />
              {form.errors.sourceKind && (
                <Text size="xs" c="red">
                  {form.errors.sourceKind}
                </Text>
              )}
            </Stack>
          )}

          <MultiSelect
            label="Available for tasks"
            description="Users can pick this model in experiments for these tasks only."
            data={(kind?.tasks ?? []).map((t) => ({ value: t, label: taskLabel(t) }))}
            disabled={!kind}
            {...form.getInputProps('tasks')}
          />

          {progress !== null && (
            <Stack gap={2}>
              <Progress value={progress * 100} animated />
              <Text size="xs" c="dimmed">
                Uploading… {Math.round(progress * 100)}%
              </Text>
            </Stack>
          )}

          <Group justify="flex-end">
            <Button variant="subtle" onClick={close} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" loading={create.isPending} disabled={!!kind?.unavailableReason}>
              {form.values.sourceKind === 'upload' ? 'Add and upload' : 'Add model'}
            </Button>
          </Group>
        </Stack>
      </form>
    </Modal>
  )
}

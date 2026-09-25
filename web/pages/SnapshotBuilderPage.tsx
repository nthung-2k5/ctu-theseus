/**
 * Snapshot builder: split the draft, optionally augment the training split, and freeze the result as
 * an immutable snapshot. Augmented copies become real train-split items of the snapshot (see
 * services/augmentation.py), so the estimate on the right is exactly `train × copies` at most.
 */

import { Alert, Badge, Button, Group, Paper, Stack, Switch, Table, Text, TextInput } from '@mantine/core'
import { useForm } from '@mantine/form'
import { notifications } from '@mantine/notifications'
import { ArrowLeftIcon, MagicWandIcon, WarningCircleIcon } from '@phosphor-icons/react'
import {
  AugmentationConfigForm,
  emptyAugmentationDraft,
  toAugmentationConfig,
} from '@public/components/dataset/AugmentationConfigForm'
import { AutoSplitForm } from '@public/components/dataset/AutoSplitForm'
import { SPLIT_TYPES, splitCounts } from '@public/components/dataset/VersionBrowsing'
import { EmptyState, LinkButton, PageHeader, ProportionBar, SectionLabel } from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import {
  getCreateVersionMutationOptions,
  getListAugmentationOptionsQueryOptions,
} from '@public/lib/api/generated/datasets/datasets'
import { SPLIT_COLORS } from '@public/lib/constants'
import { AUGMENTED_COLOR } from '@public/lib/palette'
import { invalidateProjectScope, projectDetailQueryOptions, useProjectItems } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import { useMutation, useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/snapshots/new')

const cap = (s: string) => `${s[0].toUpperCase()}${s.slice(1)}`

export function SnapshotBuilderPage() {
  const { projectId } = routeApi.useParams()
  const navigate = routeApi.useNavigate()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const draft = project.dataset?.draft
  const descriptor = getTaskDescriptor(project.task)
  const needsAnnotations = descriptor.columns.some((c) => c.kind === 'label' || c.kind === 'text_sequence_label')

  const form = useForm({
    initialValues: { versionTag: '' },
    validate: { versionTag: (v) => (v.trim().length > 0 ? null : 'Snapshot tag is required') },
  })

  // Augmentation is chosen here, at snapshot creation: the copies become real train-split items.
  const { data: augmentationOptions } = useQuery(getListAugmentationOptionsQueryOptions(projectId))
  const augmentations = augmentationOptions?.augmentations ?? []
  const [augment, setAugment] = useState(false)
  const [augmentationDraft, setAugmentationDraft] = useState(emptyAugmentationDraft)
  const augmentationConfig = augment ? toAugmentationConfig(augmentationDraft) : undefined

  // total/labeledCount are computed server-side regardless of perPage, so this is a cheap way to read them.
  const { data: draftCounts, isLoading: loadingDraftCounts } = useProjectItems(projectId, { perPage: 1 })
  const total = draftCounts?.total ?? 0
  const unlabeledCount = total - (draftCounts?.labeledCount ?? 0)
  const isEmpty = !loadingDraftCounts && total === 0

  const counts = draft ? splitCounts(draft) : { train: 0, validation: 0, test: 0 }
  const added = augmentationConfig ? counts.train * augmentationDraft.copiesPerItem : 0
  const grandTotal = counts.train + counts.validation + counts.test + added

  const createVersion = useMutation({
    ...getCreateVersionMutationOptions(),
    onSuccess: () => {
      invalidateProjectScope(queryClient, projectId)
      notifications.show({ title: 'Snapshot created', message: 'The snapshot is building', color: 'green' })
      void navigate({ to: '/project/$projectId/snapshots', params: { projectId } })
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to create snapshot'), color: 'red' })
    },
  })

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="New snapshot"
        description="Freeze the current draft as an immutable, versioned dataset for training."
        actions={
          <LinkButton
            to="/project/$projectId/snapshots"
            params={{ projectId }}
            variant="default"
            leftSection={<ArrowLeftIcon size={14} />}
          >
            Back to snapshots
          </LinkButton>
        }
      />

      {!draft ? (
        <EmptyState title="No draft dataset found" description="This project's dataset hasn't been initialized yet." />
      ) : (
        <form
          onSubmit={form.onSubmit((values) =>
            createVersion.mutate({
              projectId,
              data: { versionTag: values.versionTag, augmentation: augmentationConfig },
            }),
          )}
        >
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <Stack gap="sm">
              {/* ─── 1 · Split ─── */}
              <Paper p="md">
                <Stack gap="sm">
                  <SectionLabel>1 · Split</SectionLabel>
                  <ProportionBar
                    segments={SPLIT_TYPES.map((s) => ({
                      key: s,
                      label: cap(s),
                      value: counts[s],
                      color: SPLIT_COLORS[s] ?? 'gray',
                    }))}
                  />
                  <Group gap="md" className="tnum">
                    {SPLIT_TYPES.map((s) => (
                      <Text key={s} size="xs" c="dimmed">
                        <span
                          style={{
                            display: 'inline-block',
                            width: 8,
                            height: 8,
                            marginRight: 6,
                            background: SPLIT_COLORS[s],
                          }}
                        />
                        {cap(s)} {counts[s]}
                      </Text>
                    ))}
                  </Group>
                  <AutoSplitForm
                    projectId={projectId}
                    requiresLabelClasses={descriptor.annotation.requiresLabelClasses}
                  />
                </Stack>
              </Paper>

              {/* ─── 2 · Augmentation ─── */}
              {augmentations.length > 0 && (
                <Paper p="md">
                  <Stack gap="sm">
                    <SectionLabel>2 · Augmentation</SectionLabel>
                    <Switch
                      label="Augment the training split"
                      description="Add randomly perturbed copies of every training item. You can browse and filter them on the snapshot page."
                      checked={augment}
                      onChange={(e) => setAugment(e.currentTarget.checked)}
                    />
                    {augment && (
                      <>
                        <Alert color="blue" p="xs" icon={<MagicWandIcon size={16} />}>
                          Train only: validation and test items are never augmented, so nothing leaks into evaluation.
                        </Alert>
                        <AugmentationConfigForm
                          options={augmentations}
                          draft={augmentationDraft}
                          onChange={setAugmentationDraft}
                          trainCount={counts.train}
                        />
                      </>
                    )}
                  </Stack>
                </Paper>
              )}
            </Stack>

            {/* ─── 3 · Result (estimate) ─── */}
            <div>
              <Paper p="md" style={{ position: 'sticky', top: 'calc(48px + var(--mantine-spacing-md))' }}>
                <Stack gap="sm">
                  <SectionLabel>{augmentations.length > 0 ? '3' : '2'} · Result (estimate)</SectionLabel>

                  {isEmpty && (
                    <Alert icon={<WarningCircleIcon size={16} />} color="red" p="xs" title="The draft is empty">
                      Add items on the Upload page first; there is nothing to freeze yet.
                    </Alert>
                  )}
                  {needsAnnotations && unlabeledCount > 0 && (
                    <Alert icon={<WarningCircleIcon size={16} />} color="yellow" p="xs" title="Unlabeled items">
                      {unlabeledCount} of {total} items have no label. They snapshot with an empty label and give no
                      training signal.
                    </Alert>
                  )}

                  <Table verticalSpacing={4} withRowBorders={false} className="tnum">
                    <Table.Tbody>
                      {SPLIT_TYPES.map((s) => (
                        <Table.Tr key={s}>
                          <Table.Td>
                            <Group gap={6} wrap="nowrap">
                              <span style={{ width: 8, height: 8, background: SPLIT_COLORS[s] }} />
                              <Text size="sm">{cap(s)}</Text>
                            </Group>
                          </Table.Td>
                          <Table.Td ta="right">
                            <Text size="sm">{counts[s]}</Text>
                          </Table.Td>
                          <Table.Td ta="right">
                            {s === 'train' && added > 0 && (
                              <Text size="xs" style={{ color: AUGMENTED_COLOR }}>
                                +{added}
                              </Text>
                            )}
                          </Table.Td>
                        </Table.Tr>
                      ))}
                      <Table.Tr>
                        <Table.Td>
                          <Text size="sm" fw={600}>
                            Total
                          </Text>
                        </Table.Td>
                        <Table.Td ta="right" colSpan={2}>
                          <Group gap={6} justify="flex-end">
                            {added > 0 && counts.train > 0 && (
                              <Badge color="grape">×{(grandTotal / (grandTotal - added || 1)).toFixed(2)}</Badge>
                            )}
                            <Text size="sm" fw={600}>
                              {grandTotal}
                            </Text>
                          </Group>
                        </Table.Td>
                      </Table.Tr>
                    </Table.Tbody>
                  </Table>

                  <ProportionBar
                    segments={[
                      ...SPLIT_TYPES.map((s) => ({
                        key: s,
                        label: cap(s),
                        value: counts[s],
                        color: SPLIT_COLORS[s] ?? 'gray',
                      })),
                      { key: 'aug', label: 'Augmented', value: added, color: AUGMENTED_COLOR, dashed: true },
                    ]}
                  />

                  <TextInput
                    label="Snapshot tag"
                    placeholder="e.g. v1.0, baseline-aug"
                    {...form.getInputProps('versionTag')}
                  />
                  <Button
                    type="submit"
                    leftSection={<MagicWandIcon size={14} />}
                    loading={createVersion.isPending}
                    disabled={isEmpty || (augment && !augmentationConfig)}
                  >
                    Create snapshot
                  </Button>
                </Stack>
              </Paper>
            </div>
          </div>
        </form>
      )}
    </div>
  )
}

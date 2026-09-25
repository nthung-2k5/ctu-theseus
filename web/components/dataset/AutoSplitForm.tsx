import { Button, Checkbox, Group, NumberInput, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ShuffleIcon } from '@phosphor-icons/react'
import { autoSplitItems } from '@public/lib/api/generated/datasets/datasets'
import { invalidateProjectScope } from '@public/lib/queries'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

/**
 * Randomly reassigns every draft item to train/validation/test in a given ratio. Shared by the
 * Dataset page's modal and the snapshot builder's "Split" step.
 */
export function AutoSplitForm({
  projectId,
  requiresLabelClasses,
  onDone,
  onCancel,
}: {
  projectId: string
  requiresLabelClasses: boolean
  onDone?: () => void
  onCancel?: () => void
}) {
  const [ratios, setRatios] = useState({ train: 80, validation: 10, test: 10 })
  const [stratify, setStratify] = useState(true)
  const queryClient = useQueryClient()
  const total = ratios.train + ratios.validation + ratios.test

  const autoSplit = useMutation({
    mutationFn: async () => autoSplitItems(projectId, { ratios, stratify: requiresLabelClasses ? stratify : false }),
    onSuccess: (data) => {
      invalidateProjectScope(queryClient, projectId)
      notifications.show({
        title: 'Auto-split complete',
        message: `${data.updated} item(s) reassigned`,
        color: 'green',
      })
      for (const warning of data.warnings ?? []) {
        notifications.show({ title: 'Split warning', message: warning, color: 'yellow' })
      }
      onDone?.()
    },
    onError: () => {
      notifications.show({ title: 'Error', message: 'Auto-split failed', color: 'red' })
    },
  })

  return (
    <Stack gap="sm">
      <Text size="xs" c="dimmed">
        Randomly shuffles every item currently in the draft into train/validation/test, matching this ratio as closely
        as rounding allows. This overwrites any manual split assignments.
      </Text>
      <Group grow>
        <NumberInput
          size="xs"
          label="Train"
          min={0}
          value={ratios.train}
          onChange={(v) => setRatios((r) => ({ ...r, train: Number(v) || 0 }))}
        />
        <NumberInput
          size="xs"
          label="Validation"
          min={0}
          value={ratios.validation}
          onChange={(v) => setRatios((r) => ({ ...r, validation: Number(v) || 0 }))}
        />
        <NumberInput
          size="xs"
          label="Test"
          min={0}
          value={ratios.test}
          onChange={(v) => setRatios((r) => ({ ...r, test: Number(v) || 0 }))}
        />
      </Group>
      {total > 0 && (
        <Text size="xs" c="dimmed" className="tnum">
          {((ratios.train / total) * 100).toFixed(0)}% / {((ratios.validation / total) * 100).toFixed(0)}% /{' '}
          {((ratios.test / total) * 100).toFixed(0)}%
        </Text>
      )}
      {requiresLabelClasses && (
        <Checkbox
          size="xs"
          label="Stratify by label class"
          description="Keeps each class's items in the same ratio across train/validation/test. Recommended: an unstratified split can leave a class entirely out of validation or test."
          checked={stratify}
          onChange={(e) => setStratify(e.currentTarget.checked)}
        />
      )}
      <Group justify="flex-end">
        {onCancel && (
          <Button variant="subtle" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button
          leftSection={<ShuffleIcon size={14} />}
          disabled={total <= 0}
          loading={autoSplit.isPending}
          onClick={() => autoSplit.mutate()}
        >
          Apply split
        </Button>
      </Group>
    </Stack>
  )
}

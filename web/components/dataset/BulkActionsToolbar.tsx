import { Button, Card, Group, Menu, Text, Tooltip } from '@mantine/core'
import { CaretDownIcon, ListChecksIcon, TagIcon, TrashIcon, XIcon } from '@phosphor-icons/react'
import { confirmDelete } from '@public/components/ui'
import { useItemMutations } from '@public/hooks/useItemMutations'
import { MAX_EXCLUSIONS, MAX_FILTER_BULK, type SelectionRequest } from '@public/lib/itemSelection'
import { useLabelClasses } from '@public/lib/queries'
import { SPLIT_TYPES } from './VersionBrowsing'

/**
 * Bulk actions over the current selection: explicit ids, or "everything matching the filter" (minus any
 * ticked off), which the server resolves. `count` is how many items that covers.
 */
export const BulkActionsToolbar = ({
  projectId,
  request,
  count,
  exclusionCount,
  onClear,
  requiresLabelClasses,
  pageCount,
  total,
  onSelectPage,
  onSelectAllMatching,
}: {
  projectId: string
  request: SelectionRequest
  count: number
  exclusionCount: number
  onClear: () => void
  requiresLabelClasses: boolean
  pageCount: number
  total: number
  onSelectPage: () => void
  onSelectAllMatching: () => void
}) => {
  const { data: classesData } = useLabelClasses(requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []
  const { reassignSplit, assignClass, removeItems } = useItemMutations(projectId)

  const onSuccess = { onSuccess: onClear }
  const tooManyMatches = total > MAX_FILTER_BULK
  const tooManyExclusions = exclusionCount > MAX_EXCLUSIONS
  const blocked = tooManyExclusions
    ? `Too many items un-ticked (max ${MAX_EXCLUSIONS}). Clear and narrow the filter.`
    : null

  const handleDelete = () =>
    confirmDelete({
      title: 'Delete items',
      message: `Remove ${count} item(s) from the draft? Items still referenced by a snapshot are kept (hidden from the draft, preserved for that snapshot).`,
      onConfirm: () => removeItems.mutate({ selection: request }, onSuccess),
    })

  return (
    <Card
      withBorder
      p="xs"
      radius="md"
      shadow="lg"
      style={{
        position: 'sticky',
        bottom: '4rem',
        zIndex: 200,
      }}
    >
      <Group justify="space-between">
        <Text size="sm" fw={600}>
          {count} selected
          {blocked && (
            <Text span size="xs" c="red" ml="xs">
              {blocked}
            </Text>
          )}
        </Text>
        <Group gap="xs">
          <Menu shadow="md" position="bottom-start">
            <Menu.Target>
              <Button
                size="xs"
                variant="light"
                color="gray"
                rightSection={<CaretDownIcon size={12} />}
                leftSection={<ListChecksIcon size={12} />}
              >
                Select
              </Button>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item onClick={onSelectPage}>Select all on this page ({pageCount})</Menu.Item>
              <Menu.Item
                onClick={onSelectAllMatching}
                disabled={tooManyMatches}
                title={
                  tooManyMatches ? `Bulk actions cover at most ${MAX_FILTER_BULK} items: narrow the filter.` : undefined
                }
              >
                Select all matching items ({total})
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>

          {count > 0 && !blocked && (
            <>
              <Menu shadow="md" position="bottom-start">
                <Menu.Target>
                  <Button size="xs" variant="light" rightSection={<CaretDownIcon size={12} />}>
                    Assign split
                  </Button>
                </Menu.Target>
                <Menu.Dropdown>
                  {SPLIT_TYPES.map((s) => (
                    <Menu.Item
                      key={s}
                      tt="capitalize"
                      onClick={() => reassignSplit.mutate({ selection: request, split: s }, onSuccess)}
                    >
                      {s}
                    </Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>

              {requiresLabelClasses && classes.length > 0 && (
                <Menu shadow="md" position="bottom-start">
                  <Menu.Target>
                    <Button
                      size="xs"
                      variant="light"
                      color="violet"
                      rightSection={<CaretDownIcon size={12} />}
                      leftSection={<TagIcon size={12} />}
                    >
                      Assign class
                    </Button>
                  </Menu.Target>
                  <Menu.Dropdown>
                    {classes.map((c) => (
                      <Menu.Item
                        key={c.classId}
                        onClick={() => assignClass.mutate({ selection: request, classId: c.classId }, onSuccess)}
                      >
                        {c.name}
                      </Menu.Item>
                    ))}
                  </Menu.Dropdown>
                </Menu>
              )}

              <Button
                size="xs"
                variant="light"
                color="red"
                leftSection={<TrashIcon size={12} />}
                onClick={handleDelete}
              >
                Delete
              </Button>
            </>
          )}
          <Tooltip label="Clear selection">
            <Button
              size="xs"
              variant="subtle"
              color="gray"
              onClick={onClear}
              disabled={count === 0}
              aria-label="Clear selection"
            >
              <XIcon size={14} />
            </Button>
          </Tooltip>
        </Group>
      </Group>
    </Card>
  )
}

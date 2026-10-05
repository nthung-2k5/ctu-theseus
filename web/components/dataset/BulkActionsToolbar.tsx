import { Button, Card, Group, Menu, Text, Tooltip } from '@mantine/core'
import { CaretDownIcon, ListChecksIcon, TagIcon, TrashIcon, XIcon } from '@phosphor-icons/react'
import { confirmDelete } from '@public/components/ui'
import { useItemMutations } from '@public/hooks/useItemMutations'
import { useLabelClasses } from '@public/lib/queries'
import { SPLIT_TYPES } from './VersionBrowsing'

export const BulkActionsToolbar = ({
  projectId,
  selectedIds,
  onClear,
  requiresLabelClasses,
  pageCount,
  total,
  onSelectPage,
  onSelectAllMatching,
  selectingAll,
}: {
  projectId: string
  selectedIds: ReadonlySet<string>
  onClear: () => void
  requiresLabelClasses: boolean
  pageCount: number
  total: number
  onSelectPage: () => void
  onSelectAllMatching: () => void
  selectingAll: boolean
}) => {
  const { data: classesData } = useLabelClasses(requiresLabelClasses ? projectId : undefined)
  const classes = classesData?.classes ?? []
  const { reassignSplit, assignClass, removeItems } = useItemMutations(projectId)

  const itemIds = [...selectedIds]
  const onSuccess = { onSuccess: onClear }

  const handleDelete = () =>
    confirmDelete({
      title: 'Delete items',
      message: `Remove ${itemIds.length} item(s) from the draft? Items still referenced by a snapshot are kept (hidden from the draft, preserved for that snapshot).`,
      onConfirm: () => removeItems.mutate({ itemIds }, onSuccess),
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
          {selectedIds.size} selected
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
                loading={selectingAll}
              >
                Select
              </Button>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item onClick={onSelectPage}>Select all on this page ({pageCount})</Menu.Item>
              <Menu.Item onClick={onSelectAllMatching}>Select all matching items ({total})</Menu.Item>
            </Menu.Dropdown>
          </Menu>

          {selectedIds.size > 0 && (
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
                      onClick={() => reassignSplit.mutate({ itemIds, split: s }, onSuccess)}
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
                        onClick={() => assignClass.mutate({ itemIds, classId: c.classId }, onSuccess)}
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
              disabled={selectedIds.size === 0}
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

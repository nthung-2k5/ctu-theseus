import { ActionIcon, Menu } from '@mantine/core'
import { ArrowsMergeIcon, DotsThreeVerticalIcon, ListMagnifyingGlassIcon, TrashIcon } from '@phosphor-icons/react'
import type { MergeTarget } from './RemoveClassDialog'

/** The per-class "..." menu: see its items, merge it into another class, or remove it. */
export function ClassRowMenu({
  name,
  canViewItems,
  mergeTargets,
  onViewItems,
  onMerge,
  onRemove,
}: {
  name: string
  /** A class that has never been saved has no items to look at. */
  canViewItems: boolean
  /** Other saved classes this one's items could move into; empty hides the merge entry. */
  mergeTargets: MergeTarget[]
  onViewItems: () => void
  onMerge: (targetClassId: string) => void
  onRemove: () => void
}) {
  return (
    <Menu shadow="md" position="bottom-end" withinPortal>
      <Menu.Target>
        <ActionIcon variant="subtle" color="gray" aria-label={`Actions for ${name}`}>
          <DotsThreeVerticalIcon size={16} weight="bold" />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        {canViewItems && (
          <Menu.Item leftSection={<ListMagnifyingGlassIcon size={14} />} onClick={onViewItems}>
            View items
          </Menu.Item>
        )}
        {mergeTargets.length > 0 && (
          <Menu.Sub>
            <Menu.Sub.Target>
              <Menu.Sub.Item leftSection={<ArrowsMergeIcon size={14} />}>Merge into…</Menu.Sub.Item>
            </Menu.Sub.Target>
            <Menu.Sub.Dropdown mah={280} style={{ overflowY: 'auto' }}>
              {mergeTargets.map((t) => (
                <Menu.Item key={t.classId} onClick={() => onMerge(t.classId)}>
                  {t.name}
                </Menu.Item>
              ))}
            </Menu.Sub.Dropdown>
          </Menu.Sub>
        )}
        <Menu.Divider />
        <Menu.Item color="red" leftSection={<TrashIcon size={14} />} onClick={onRemove}>
          Remove
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  )
}

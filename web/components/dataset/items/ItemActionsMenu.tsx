import { ActionIcon, Menu } from '@mantine/core'
import { ArrowSquareOutIcon, DotsThreeVerticalIcon, TagIcon, TrashIcon } from '@phosphor-icons/react'
import { confirmDelete } from '@public/components/ui'
import { SPLIT_TYPES } from '../VersionBrowsing'
import type { DatasetListItem, ItemEditing } from './types'

/** The per-item "..." menu on a card or table row: open the drawer, move to a split, relabel, delete. */
export const ItemActionsMenu = ({
  item,
  editing,
  onOpen,
}: {
  item: DatasetListItem
  editing: ItemEditing
  onOpen: () => void
}) => (
  <Menu shadow="md" position="bottom-end" withinPortal>
    <Menu.Target>
      <ActionIcon variant="default" size="sm" aria-label={`Actions for ${item.externalId ?? 'item'}`}>
        <DotsThreeVerticalIcon size={14} weight="bold" />
      </ActionIcon>
    </Menu.Target>
    <Menu.Dropdown>
      <Menu.Item leftSection={<ArrowSquareOutIcon size={14} />} onClick={onOpen}>
        Open details
      </Menu.Item>

      <Menu.Sub>
        <Menu.Sub.Target>
          <Menu.Sub.Item>Move to split</Menu.Sub.Item>
        </Menu.Sub.Target>
        <Menu.Sub.Dropdown>
          {SPLIT_TYPES.map((split) => (
            <Menu.Item
              key={split}
              tt="capitalize"
              disabled={split === item.splitType}
              onClick={() => editing.onSetSplit(item.id, split)}
            >
              {split}
            </Menu.Item>
          ))}
        </Menu.Sub.Dropdown>
      </Menu.Sub>

      {editing.classes.length > 0 && (
        <Menu.Sub>
          <Menu.Sub.Target>
            <Menu.Sub.Item leftSection={<TagIcon size={14} />}>Assign class</Menu.Sub.Item>
          </Menu.Sub.Target>
          <Menu.Sub.Dropdown>
            {editing.classes.map((c) => (
              <Menu.Item key={c.classId} onClick={() => editing.onSetClass(item.id, c.classId)}>
                {c.name}
              </Menu.Item>
            ))}
          </Menu.Sub.Dropdown>
        </Menu.Sub>
      )}

      <Menu.Divider />
      <Menu.Item
        color="red"
        leftSection={<TrashIcon size={14} />}
        onClick={() =>
          confirmDelete({
            title: 'Delete item',
            message: `Remove ${item.externalId ?? 'this item'} from the draft? An item still referenced by a snapshot is kept for that snapshot.`,
            onConfirm: () => editing.onDelete(item.id),
          })
        }
      >
        Delete
      </Menu.Item>
    </Menu.Dropdown>
  </Menu>
)

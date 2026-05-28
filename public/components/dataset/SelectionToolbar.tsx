import {
  Badge,
  Box,
  Button,
  ColorSwatch,
  Combobox,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  useCombobox,
} from '@mantine/core'
import { modals } from '@mantine/modals'
import { BrainIcon, ShieldCheckIcon, TestTubeIcon, TrashIcon } from '@phosphor-icons/react'
import type { DatasetSplitValue } from '@public/store/types'

interface SelectionToolbarProps {
  count: number
  classes: { id: string; name: string; color: string }[]
  splitStats: { train: number; validation: number; test: number }
  onAssignClass: (classId: string | null) => void
  onChangeSplit: (split: DatasetSplitValue) => void
  onDelete: () => void
  onDeselect: () => void
}

export function SelectionToolbar({
  count,
  classes,
  splitStats,
  onAssignClass,
  onChangeSplit,
  onDelete,
  onDeselect,
}: SelectionToolbarProps) {
  const classCombobox = useCombobox({ onDropdownClose: () => classCombobox.resetSelectedOption() })

  const openSplitModal = () => {
    modals.open({
      title: 'Change dataset split',
      centered: true,
      size: 'lg',
      children: (
        <SimpleGrid cols={3} spacing="md">
          {/* Train */}
          <Stack align="center" gap="sm" p="md">
            <Box
              style={{
                width: 40,
                height: 40,
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--mantine-color-green-1)',
                color: 'var(--mantine-color-green-7)',
              }}
            >
              <BrainIcon size={20} weight="bold" />
            </Box>
            <Text fw={600} size="sm">
              Train
            </Text>
            <Text size="xs" c="dimmed">
              {splitStats.train} images
            </Text>
            <Button
              size="xs"
              color="green"
              variant="light"
              fullWidth
              onClick={() => {
                onChangeSplit('train')
                modals.closeAll()
              }}
            >
              Set to Train
            </Button>
          </Stack>
          {/* Validation */}
          <Stack align="center" gap="sm" p="md">
            <Box
              style={{
                width: 40,
                height: 40,
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--mantine-color-blue-1)',
                color: 'var(--mantine-color-blue-7)',
              }}
            >
              <ShieldCheckIcon size={20} weight="bold" />
            </Box>
            <Text fw={600} size="sm">
              Validation
            </Text>
            <Text size="xs" c="dimmed">
              {splitStats.validation} images
            </Text>
            <Button
              size="xs"
              color="blue"
              variant="light"
              fullWidth
              onClick={() => {
                onChangeSplit('validation')
                modals.closeAll()
              }}
            >
              Set to Validation
            </Button>
          </Stack>
          {/* Test */}
          <Stack align="center" gap="sm" p="md">
            <Box
              style={{
                width: 40,
                height: 40,
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--mantine-color-red-1)',
                color: 'var(--mantine-color-red-7)',
              }}
            >
              <TestTubeIcon size={20} weight="bold" />
            </Box>
            <Text fw={600} size="sm">
              Test
            </Text>
            <Text size="xs" c="dimmed">
              {splitStats.test} images
            </Text>
            <Button
              size="xs"
              color="red"
              variant="light"
              fullWidth
              onClick={() => {
                onChangeSplit('test')
                modals.closeAll()
              }}
            >
              Set to Test
            </Button>
          </Stack>
        </SimpleGrid>
      ),
    })
  }

  return (
    <Paper
      radius="md"
      p="sm"
      withBorder
      pos="sticky"
      bottom={16}
      bg="var(--mantine-color-dark-7)"
      style={{
        zIndex: 100,
        borderColor: 'var(--mantine-primary-color-5)',
      }}
    >
      <Group justify="space-between">
        <Group gap="sm">
          <Badge variant="filled" size="lg">
            {count} selected
          </Badge>
          <Button size="xs" variant="subtle" onClick={onDeselect}>
            Deselect
          </Button>
        </Group>
        <Group gap="xs">
          {/* Assign class combobox */}
          <Combobox
            store={classCombobox}
            onOptionSubmit={(val) => {
              onAssignClass(val === '__none__' ? null : val)
              classCombobox.closeDropdown()
            }}
          >
            <Combobox.Target>
              <Button size="xs" variant="light" onClick={() => classCombobox.toggleDropdown()}>
                Assign class
              </Button>
            </Combobox.Target>
            <Combobox.Dropdown>
              <Combobox.Options>
                <Combobox.Option value="__none__">
                  <Text size="xs" c="dimmed">
                    Unclassified
                  </Text>
                </Combobox.Option>
                {classes.map((c) => (
                  <Combobox.Option key={c.id} value={c.id}>
                    <Group gap="xs">
                      <ColorSwatch color={c.color} size={12} />
                      <Text size="xs">{c.name}</Text>
                    </Group>
                  </Combobox.Option>
                ))}
              </Combobox.Options>
            </Combobox.Dropdown>
          </Combobox>

          {/* Change split modal */}
          <Button size="xs" variant="light" onClick={openSplitModal}>
            Change split
          </Button>

          {/* Delete */}
          <Button size="xs" variant="light" color="red" onClick={onDelete}>
            <TrashIcon size={14} />
          </Button>
        </Group>
      </Group>
    </Paper>
  )
}

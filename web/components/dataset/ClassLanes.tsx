/**
 * The spot-check view: one lane per class, each a wall of small previews, so a picture in the wrong folder shows
 * up by eye (a dog under "cats") instead of by reading file names.
 *
 * A lane shows an even sample of its files first and grows on request, so a class of ten thousand photos costs a
 * dozen decoded previews, not ten thousand. Dragging previews between lanes reclassifies them; dropping files from
 * the desktop onto a lane stages them straight into that class. Nothing is sent from here.
 */

import { ActionIcon, Badge, Button, Group, Menu, NativeSelect, Paper, Stack, Text } from '@mantine/core'
import {
  CheckIcon,
  DotsThreeVerticalIcon,
  FolderSimplePlusIcon,
  LinkIcon,
  PencilSimpleIcon,
  TrashIcon,
  UploadSimpleIcon,
  WarningIcon,
  XIcon,
} from '@phosphor-icons/react'
import { ProportionBar } from '@public/components/ui'
import { SPLIT_COLORS } from '@public/lib/constants'
import { hasBlockingError, NO_CLASS_KEY } from '@public/lib/upload/fileTree'
import { buildLanes, type Lane, laneTiles } from '@public/lib/upload/lanes'
import { mediaOf } from '@public/lib/upload/review'
import type { DropContext, StagedFile } from '@public/lib/upload/types'
import type { FileStaging } from '@public/lib/upload/useFileStaging'
import type { LabelClass } from '@public/store/types'
import { type DragEvent, useMemo, useState } from 'react'
import { openMapToClass, openRenameFolder } from './FolderModals'
import { laneColor } from './laneColor'
import { Thumb } from './Thumb'

/** Carries the ids of the previews being dragged from one lane to another. */
const STAGED_MIME = 'application/x-omnitrain-staged'
/** Previews a lane shows before it is expanded, and how many more each "show more" adds. */
const SAMPLE = 12
const PAGE = 48

const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer.types).includes('Files')
const isStagedDrag = (event: DragEvent) => Array.from(event.dataTransfer.types).includes(STAGED_MIME)

function Tile({
  file,
  selected,
  locked,
  onToggle,
  onDragStart,
}: {
  file: StagedFile
  selected: boolean
  locked: boolean
  onToggle: () => void
  onDragStart: (event: DragEvent) => void
}) {
  const broken = hasBlockingError(file)
  const note = file.issues.find((i) => i.severity === 'error')?.message ?? file.skip
  const media = mediaOf(file.file.type)
  return (
    <button
      type="button"
      draggable={!locked}
      onDragStart={onDragStart}
      onClick={onToggle}
      disabled={locked}
      aria-pressed={selected}
      aria-label={`${file.name}${note ? ` — ${note}` : ''}`}
      title={note ? `${file.name} — ${note}` : file.name}
      style={{
        position: 'relative',
        aspectRatio: '1',
        padding: 0,
        border: `2px solid ${
          selected
            ? 'var(--mantine-primary-color-filled)'
            : broken
              ? 'var(--mantine-color-red-6)'
              : file.skip
                ? 'var(--mantine-color-default-border)'
                : 'transparent'
        }`,
        borderRadius: 'var(--mantine-radius-sm)',
        background: 'none',
        cursor: locked ? 'default' : 'grab',
        opacity: file.skip || broken ? 0.55 : 1,
        overflow: 'hidden',
      }}
    >
      <Thumb file={!broken && media === 'image' ? file.file : undefined} media={media} />
      {selected && (
        <span
          style={{
            position: 'absolute',
            top: 3,
            left: 3,
            width: 16,
            height: 16,
            borderRadius: '50%',
            background: 'var(--mantine-primary-color-filled)',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <CheckIcon size={10} weight="bold" />
        </span>
      )}
      {broken && (
        <span style={{ position: 'absolute', top: 3, right: 3, display: 'flex' }}>
          <WarningIcon size={16} weight="fill" color="var(--mantine-color-red-6)" />
        </span>
      )}
      {!broken && file.skip && (
        <Badge size="xs" variant="filled" color="gray" style={{ position: 'absolute', bottom: 3, left: 3 }}>
          not sent
        </Badge>
      )}
    </button>
  )
}

export function ClassLanes({
  staging,
  classes,
  taskUsesClasses,
  locked,
  openPicker,
  onDropFiles,
}: {
  staging: FileStaging
  classes: LabelClass[]
  taskUsesClasses: boolean
  locked: boolean
  openPicker: (kind: 'files' | 'folder', drop: DropContext) => void
  onDropFiles: (dataTransfer: DataTransfer, drop: DropContext) => void
}) {
  const { index, move, remove, renameFolder, mapFolder } = staging
  const lanes = useMemo(() => buildLanes(index.folders), [index.folders])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  /** Per lane: `null` = the even sample, a number = the first N files. */
  const [limits, setLimits] = useState<Record<string, number | null>>({})
  const [over, setOver] = useState<string | null>(null)

  const live = useMemo(() => {
    if (selected.size === 0) return []
    const ids = new Set(lanes.flatMap((l) => l.files.map((f) => f.id)))
    return [...selected].filter((id) => ids.has(id))
  }, [selected, lanes])

  const existingOptions = useMemo(() => classes.map((c) => ({ value: `c:${c.classId}`, label: c.name })), [classes])
  const moveTargets = useMemo(
    () =>
      lanes
        .filter((l) => !l.isNone)
        .map((l) => ({ value: l.key, label: l.pending ? `${l.name} (new class)` : l.name })),
    [lanes],
  )

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (!next.delete(id)) next.add(id)
      return next
    })

  const startDrag = (event: DragEvent, id: string) => {
    // Dragging a selected preview takes the whole selection; dragging any other takes just itself.
    const ids = selected.has(id) ? live : [id]
    event.dataTransfer.setData(STAGED_MIME, JSON.stringify(ids))
    event.dataTransfer.effectAllowed = 'move'
  }

  const laneDrop = (lane: Lane) => {
    const drop: DropContext = { classKey: lane.isNone ? null : lane.key }
    return {
      onDragOver: (event: DragEvent) => {
        if (locked || !(isStagedDrag(event) || hasFiles(event))) return
        event.preventDefault()
        event.stopPropagation()
        event.dataTransfer.dropEffect = isStagedDrag(event) ? 'move' : 'copy'
        setOver(lane.key)
      },
      onDrop: (event: DragEvent) => {
        if (locked) return
        event.preventDefault()
        event.stopPropagation()
        setOver(null)
        if (isStagedDrag(event)) {
          try {
            const ids = JSON.parse(event.dataTransfer.getData(STAGED_MIME)) as string[]
            move(ids, { classKey: drop.classKey })
          } catch {
            // a drag from somewhere else that happened to carry the type: nothing to move
          }
        } else if (hasFiles(event)) {
          onDropFiles(event.dataTransfer, drop)
        }
      },
    }
  }

  return (
    <Stack gap="xs" style={{ flex: 1, minHeight: 0 }}>
      {live.length > 0 && (
        <Group gap="xs" wrap="wrap">
          <Badge variant="light">{live.length.toLocaleString()} selected</Badge>
          {!locked && (
            <>
              {taskUsesClasses && (
                <NativeSelect
                  size="xs"
                  aria-label="Move the selected to a class"
                  data={[
                    { value: '', label: 'Move to class…' },
                    { value: NO_CLASS_KEY, label: 'No class' },
                    ...moveTargets,
                  ]}
                  value=""
                  onChange={(event) => {
                    const value = event.currentTarget.value
                    if (!value) return
                    move(live, { classKey: value === NO_CLASS_KEY ? null : value })
                    setSelected(new Set())
                  }}
                />
              )}
              <Button
                size="xs"
                variant="light"
                color="red"
                leftSection={<TrashIcon size={14} />}
                onClick={() => {
                  remove(live)
                  setSelected(new Set())
                }}
              >
                Remove
              </Button>
            </>
          )}
          <Button size="xs" variant="subtle" color="gray" onClick={() => setSelected(new Set())}>
            Clear selection
          </Button>
        </Group>
      )}

      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          alignContent: 'start',
          gap: 12,
          padding: 2,
        }}
      >
        {lanes.map((lane) => {
          const limit = limits[lane.key] ?? null
          const tiles = laneTiles(lane.files, limit, SAMPLE)
          const total = lane.files.length
          const unassigned = lane.isNone && taskUsesClasses && total > 0
          const active = over === lane.key
          const segments = [
            ...(['train', 'validation', 'test'] as const).map((split) => ({
              key: split,
              label: split,
              value: lane.bySplit[split],
              color: SPLIT_COLORS[split],
            })),
            { key: 'auto', label: 'auto-split', value: lane.bySplit.auto, color: 'var(--mantine-color-gray-6)' },
          ]
          return (
            <Paper
              key={lane.key}
              withBorder
              p="xs"
              radius="md"
              {...laneDrop(lane)}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOver(null)
              }}
              style={{
                borderColor: active
                  ? 'var(--mantine-primary-color-filled)'
                  : unassigned
                    ? 'var(--mantine-color-yellow-6)'
                    : undefined,
                borderStyle: active || total === 0 ? 'dashed' : 'solid',
                background: active ? 'var(--mantine-primary-color-light)' : undefined,
              }}
            >
              <Stack gap={6}>
                <Group justify="space-between" wrap="nowrap" gap="xs">
                  <Group gap={6} wrap="nowrap" style={{ minWidth: 0 }}>
                    <span
                      style={{ width: 10, height: 10, borderRadius: '50%', background: laneColor(lane), flexShrink: 0 }}
                    />
                    <Text size="sm" fw={600} truncate>
                      {lane.isNone && taskUsesClasses ? 'No class yet' : lane.name}
                    </Text>
                    {lane.pending && (
                      <Badge size="xs" variant="light" color="teal">
                        new class
                      </Badge>
                    )}
                  </Group>
                  <Group gap={4} wrap="nowrap">
                    <Text size="xs" c="dimmed">
                      {total.toLocaleString()} file{total === 1 ? '' : 's'}
                    </Text>
                    {!locked && (
                      <Menu position="bottom-end" withinPortal>
                        <Menu.Target>
                          <ActionIcon size="sm" variant="subtle" color="gray" aria-label={`Actions for ${lane.name}`}>
                            <DotsThreeVerticalIcon size={16} />
                          </ActionIcon>
                        </Menu.Target>
                        <Menu.Dropdown>
                          <Menu.Item
                            leftSection={<UploadSimpleIcon size={14} />}
                            onClick={() => openPicker('files', { classKey: lane.isNone ? null : lane.key })}
                          >
                            Add files here…
                          </Menu.Item>
                          <Menu.Item
                            leftSection={<FolderSimplePlusIcon size={14} />}
                            onClick={() => openPicker('folder', { classKey: lane.isNone ? null : lane.key })}
                          >
                            Add folder here…
                          </Menu.Item>
                          {lane.pending && (
                            <>
                              <Menu.Item
                                leftSection={<PencilSimpleIcon size={14} />}
                                onClick={() => openRenameFolder(lane.name, (name) => renameFolder(lane.key, name))}
                              >
                                Rename new class…
                              </Menu.Item>
                              <Menu.Item
                                leftSection={<LinkIcon size={14} />}
                                onClick={() =>
                                  openMapToClass(lane.name, existingOptions, (classKey) =>
                                    mapFolder(lane.key, classKey),
                                  )
                                }
                              >
                                Use an existing class…
                              </Menu.Item>
                            </>
                          )}
                          {total > 0 && (
                            <>
                              <Menu.Divider />
                              <Menu.Item
                                color="red"
                                leftSection={<TrashIcon size={14} />}
                                onClick={() => remove(lane.files.map((f) => f.id))}
                              >
                                Remove all files here
                              </Menu.Item>
                            </>
                          )}
                        </Menu.Dropdown>
                      </Menu>
                    )}
                  </Group>
                </Group>

                {total > 0 ? (
                  <>
                    <ProportionBar segments={segments} height={6} />
                    <div
                      style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(64px, 1fr))', gap: 4 }}
                    >
                      {tiles.map((file) => (
                        <Tile
                          key={file.id}
                          file={file}
                          selected={selected.has(file.id)}
                          locked={locked}
                          onToggle={() => toggle(file.id)}
                          onDragStart={(event) => startDrag(event, file.id)}
                        />
                      ))}
                    </div>
                    <Group justify="space-between" gap="xs" wrap="nowrap">
                      <Text size="xs" c="dimmed">
                        {tiles.length < total
                          ? limit === null
                            ? `${tiles.length} of ${total.toLocaleString()}, spread across the folder`
                            : `${tiles.length.toLocaleString()} of ${total.toLocaleString()}`
                          : 'all shown'}
                      </Text>
                      <Group gap={2} wrap="nowrap">
                        {tiles.length < total && (
                          <Button
                            size="compact-xs"
                            variant="subtle"
                            onClick={() => setLimits((l) => ({ ...l, [lane.key]: (limit ?? 0) + PAGE }))}
                          >
                            Show more
                          </Button>
                        )}
                        {limit !== null && (
                          <ActionIcon
                            size="sm"
                            variant="subtle"
                            color="gray"
                            aria-label="Back to a sample"
                            onClick={() => setLimits((l) => ({ ...l, [lane.key]: null }))}
                          >
                            <XIcon size={12} />
                          </ActionIcon>
                        )}
                      </Group>
                    </Group>
                  </>
                ) : (
                  <Text size="xs" c="dimmed" ta="center" py="md">
                    Nothing here yet — drop files or drag pictures onto this class
                  </Text>
                )}
                {unassigned && (
                  <Text size="xs" c="yellow.7">
                    These have no class. Drag them onto a class, or select them and use "Move to class".
                  </Text>
                )}
              </Stack>
            </Paper>
          )
        })}
      </div>
    </Stack>
  )
}

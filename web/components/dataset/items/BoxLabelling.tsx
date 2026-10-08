import { ActionIcon, Box, Button, Group, Select, Stack, Text } from '@mantine/core'
import { TrashIcon } from '@phosphor-icons/react'
import { type Box as BoxGeometry, backgroundMarker, boxesOf, classColor, nudgeBox } from '@public/lib/boxes'
import { useRef, useState } from 'react'
import { BoxCanvas } from './BoxCanvas'
import type { BoxEditing, DatasetListItem } from './types'

/** Arrow-key nudges are applied at once on screen and saved when the key presses stop, not once per press. */
const NUDGE_SAVE_DELAY_MS = 400

const Dot = ({ color }: { color: string }) => (
  <span
    aria-hidden
    style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }}
  />
)

/**
 * An image item of an object-detection project: the picture with its boxes, and, with `editing`, the controls to
 * draw, reclass, move and delete them. Without `editing` (a snapshot is immutable) it is a read-only view.
 *
 * Key it by item id, like the other editors: the selection and the unsaved nudges belong to one item.
 */
export const BoxLabelling = ({
  item,
  classNameById,
  classColorById,
  editing,
}: {
  item: DatasetListItem
  classNameById: Map<string, string>
  classColorById?: Map<string, string>
  editing?: BoxEditing
}) => {
  const boxes = boxesOf(item.annotations)
  const marker = backgroundMarker(item.annotations)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [drawClassId, setDrawClassId] = useState<string | null>(editing?.classes[0]?.classId ?? null)
  // A box nudged on screen but not saved yet.
  const [pending, setPending] = useState<Record<string, BoxGeometry>>({})
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  const classes = editing?.classes ?? []
  const canEdit = editing !== undefined && classes.length > 0
  // The classes' own colours win; `classColorById` covers a read-only view, which has no class list to read them from.
  const ownColors = new Map(classColorById)
  for (const c of classes) if (c.uiColorHex) ownColors.set(c.classId, c.uiColorHex)
  const colorOf = (classId: string | null) => classColor(classId, ownColors)
  const drawColor = colorOf(drawClassId)

  const nudge = (id: string, dx: number, dy: number) => {
    const current = pending[id] ?? boxes.find((b) => b.id === id)?.box
    if (!current || !editing) return
    const next = nudgeBox(current, dx, dy)
    setPending((p) => ({ ...p, [id]: next }))
    clearTimeout(timers.current[id])
    timers.current[id] = setTimeout(async () => {
      try {
        await editing.onUpdate(id, { box: next })
      } finally {
        setPending(({ [id]: _saved, ...rest }) => rest)
      }
    }, NUDGE_SAVE_DELAY_MS)
  }

  const remove = async (id: string) => {
    if (selectedId === id) setSelectedId(null)
    await editing?.onRemove(id)
  }

  return (
    <Stack gap="sm">
      <BoxCanvas
        src={item.downloadUrl}
        alt={item.externalId ?? 'Image'}
        drawColor={drawColor}
        boxes={boxes.map((b) => ({
          id: b.id,
          box: pending[b.id] ?? b.box,
          color: colorOf(b.classId),
          label: classNameById.get(b.classId ?? ''),
          selected: b.id === selectedId,
        }))}
        onDraw={
          canEdit && drawClassId
            ? (box) => {
                setSelectedId(null)
                void editing.onAdd(item.id, drawClassId, box)
              }
            : undefined
        }
        onSelect={setSelectedId}
        onNudge={nudge}
        onDelete={remove}
      />

      {editing && classes.length === 0 && (
        <Text size="xs" c="dimmed">
          Add at least one class on the Classes page, then draw boxes here.
        </Text>
      )}

      {canEdit && (
        <>
          <Select
            label="Draw as"
            size="xs"
            data={classes.map((c) => ({ value: c.classId, label: c.name }))}
            value={drawClassId}
            allowDeselect={false}
            leftSection={<Dot color={drawColor} />}
            onChange={setDrawClassId}
          />
          <Text size="xs" c="dimmed">
            Drag on the image to draw a box. Click a box to select it: Delete removes it and the arrow keys move it.
          </Text>
        </>
      )}

      {boxes.length > 0 ? (
        <Stack gap={4}>
          {boxes.map((b) => (
            <Group
              key={b.id}
              gap="xs"
              wrap="nowrap"
              p={4}
              style={{
                borderRadius: 6,
                outline: b.id === selectedId ? `2px solid ${colorOf(b.classId)}` : undefined,
                cursor: 'pointer',
              }}
              onClick={() => setSelectedId(b.id === selectedId ? null : b.id)}
            >
              <Dot color={colorOf(b.classId)} />
              {canEdit ? (
                // Stop the click here so changing the class does not also toggle the row's selection.
                <Box style={{ flex: 1 }} onClick={(e) => e.stopPropagation()}>
                  <Select
                    size="xs"
                    aria-label="Class of this box"
                    data={classes.map((c) => ({ value: c.classId, label: c.name }))}
                    value={b.classId}
                    allowDeselect={false}
                    onChange={(next) => next && next !== b.classId && editing.onUpdate(b.id, { classId: next })}
                  />
                </Box>
              ) : (
                <Text size="sm" style={{ flex: 1 }}>
                  {classNameById.get(b.classId ?? '') ?? 'Unknown class'}
                </Text>
              )}
              {canEdit && (
                <ActionIcon
                  variant="subtle"
                  color="red"
                  size="sm"
                  aria-label="Delete this box"
                  onClick={(e) => {
                    e.stopPropagation()
                    void remove(b.id)
                  }}
                >
                  <TrashIcon size={14} />
                </ActionIcon>
              )}
            </Group>
          ))}
        </Stack>
      ) : marker ? (
        <Group justify="space-between">
          <Text size="sm">Marked as having no objects</Text>
          {canEdit && (
            <Button size="compact-xs" variant="subtle" onClick={() => void editing.onRemove(marker.id)}>
              Clear mark
            </Button>
          )}
        </Group>
      ) : (
        <Group justify="space-between">
          <Text size="sm" c="dimmed">
            Not labelled yet
          </Text>
          {canEdit && (
            <Button size="compact-xs" variant="light" onClick={() => void editing.onMarkEmpty(item.id)}>
              Mark as having no objects
            </Button>
          )}
        </Group>
      )}
    </Stack>
  )
}

# Tasks

A task is the **data side** of a machine-learning problem: what a dataset item looks like, its ground truth, the
columns a snapshot writes, and what a trained model predicts. It says nothing about how the task is trained; that is
a [trainer backend](../plugins/backends/README.md)'s job. One `Task` subclass per task, in [`builtin/`](builtin/).

`projects.task` is free text (a task id) and `POST /api/projects` validates it against the registry, so a new task is
one class: no enum, migration or frontend edit. The picker and per-modality UI read it from
`schema/task_registry.json`.

## How a task is found

`Registry` imports every module directly under `omnitrain.tasks.builtin` whose name does not start with `_`. A
`Task` subclass registers itself when created, but only if it sets its **own** `id`. A subclass without an `id` is a
helper base (see `builtin/_common.py`). A task missing a required field raises `TypeError` at import time.
`domain/task_registry.py` is the framework-neutral lookup over the registry (`get_task_descriptor`,
`TASK_REGISTRY`, ...).

## Adding a task

1. **Add a class** to a module in `builtin/` (an existing one for its modality, or a new `builtin/<name>.py`).
2. **Set the fields** (all `ClassVar`):

   | Field | Meaning |
   | --- | --- |
   | `id` | Stable string stored in `projects.task`. |
   | `label` | Name in the picker. |
   | `modality` | `"vision"`, `"text"`, `"audio"`, `"video"` or `"tabular"`. Closed vocabulary. |
   | `item_spec` | `ItemSpec(payload="file" \| "inline_text" \| "record", accept=[...mime types])`. |
   | `annotation` | `AnnotationSpec(type=..., requires_label_classes=...)`. `type` is a closed vocabulary. |
   | `columns` | The snapshot parquet columns, in order, as `ColumnSpec`s. Required unless `planned`. |
   | `output` | `TaskOutput(column=..., kind=...)`: what a model predicts. Required unless `planned`. |
   | `input_fields` | For inline-text tasks, the input field names in the order the model expects. |
   | `status` | `"stable"`, `"experimental"` (default) or `"planned"`. A `planned` task has no `columns` or `output`. |
   | `order` | Picker order, ascending. The built-ins use 10, 20, ...; the default of 1000 sorts after them. |

3. **Prefer a helper base** from `builtin/_common.py` when the shape fits; it fills `columns` and `output` for you:
   `ClassificationTask` (set `source_column`), `LlmTextTask` (set `input_fields` and `output_field`),
   `FileCaptionTask` and `PlannedTask`.
4. **Regenerate the shared schema** with `bun run schema` from the repo root (it runs `server/scripts/export_schema.py`
   and writes `schema/task_registry.json`). CI fails if it is stale (`bun run schema:check`).
5. **Optionally add a picker icon and blurb** in `web/lib/task-meta.ts`. A task it has not heard of falls back to a
   generic one.
6. **Add a backend that supports it.** A task is only trainable once some backend's `supports(task)` accepts it
   (`backends.registry.trainable_backends`). Until then it exists but cannot be trained.
7. **Test** it like `server/tests/test_task_registry.py`.

## Minimal example

```python
# builtin/vision.py (a one-class-per-image task; the helper fills in columns and output)
from omnitrain.tasks.base import ItemSpec
from omnitrain.tasks.builtin._common import IMAGE_MIME, IMAGE_PATH, ClassificationTask, col


class LeafDiseaseClassification(ClassificationTask):
    id = "leaf_disease_classification"
    label = "Leaf Disease Classification"
    modality = "vision"
    status = "experimental"
    order = 35
    item_spec = ItemSpec(payload="file", accept=IMAGE_MIME)
    source_column = col(IMAGE_PATH, "storage_uri")
```

A free-text task is just as short:

```python
class Paraphrasing(LlmTextTask):
    id = "paraphrasing"
    label = "Paraphrasing"
    order = 110
    input_fields = ["sentence"]
    output_field = "paraphrase"
```

## What stays a code change

`modality`, `annotation.type`, column kinds and output kinds are closed on purpose: preprocessing and augmentation
ops, annotation editors and the snapshot builder are keyed on them. A task that needs a new value there is a code
change in those places, plus an `ALTER TYPE ... ADD VALUE` migration where the value is stored in a Postgres enum
(see migration `0013`, which added `video`).

## Rules that are easy to break

- **Several places assume one class per item.** Item filters, class stats, merges and the health report must go
  through `task_registry.class_annotation_type(task)`, not hard-code `classification`. Bounding boxes are one
  `bounding_box` annotation each (validated in `domain/boxes.py`), so do not re-derive those rules.
- **Do not put training concepts here.** Encoders, feature types, model families and the like belong to a backend.
- **A `planned` task must not define `columns` or `output`**, and a non-planned one must define both.

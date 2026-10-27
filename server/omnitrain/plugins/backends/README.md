# Trainer backends

One package per training framework. Ludwig (`ludwig/`), Ultralytics YOLO (`yolo/`), scikit-learn (`scikit/`),
torchvision video (`video/`), a code-defined PyTorch backend (`pytorch/`: image, audio and text
classification, tabular classification and regression, token classification) and a language-model backend (`llm/`: LoRA
fine-tuning of a bring-your-own causal LM for the four generative text tasks) ship today. A backend decides *how* a task is trained; *what* a task is lives in
[`omnitrain/tasks/`](../../tasks/README.md).

Nothing else needs an edit to add one: no enum, no migration, no frontend change. `training_runs.backend` and
`sweeps.backend` are free text holding the plugin id. Restart the API and `GET /api/training-backends` lists the
backend with its models and hyperparameters, and the web renders them.

## How a backend is found

`Registry` (`core/plugins.py`) imports every module or package directly under `omnitrain.plugins.backends` whose
name does not start with `_`, the first time anything looks one up. A `TrainerBackend` subclass registers itself
when the class is created, but only if it sets its **own** `id`. A subclass that does not set `id` is a helper base
and is ignored. A duplicate `id` raises at import time.

## Adding a backend

1. **Create the package** `plugins/backends/<name>/__init__.py` (a single module `<name>.py` also works).
2. **Subclass `TrainerBackend`** (`base.py`) and set `id` and `label`. `description` is shown in the UI.
3. **Declare `Hyperparameters`**: a subclass of `HyperparamsBase` (camelCase on the wire, unknown keys rejected).
   Add only the knobs your framework understands. The create-run form is generated from this model, so use
   `Field(..., title=, description=, ge=, le=)`. Override `hyperparameter_specs(task)` only when defaults or bounds
   differ per task.
4. **Implement the abstract methods**, all classmethods:

   | Method | What it does |
   | --- | --- |
   | `supports(task)` | Whether this backend can train the task at all. Usually a lookup of `task.id`. |
   | `models(task)` | The `ModelChoice` list for the picker. Empty is valid for one fixed model. |
   | `compile(task, ctx, hp, custom=None)` | Task + snapshot context + validated hyperparameters to your own config dict. It is stored verbatim as `training_runs.config`. Raise `ConfigError` for a problem the user can fix. |
   | `train(run)` | Train synchronously and return a `LoadedModel`. See "The training contract" below. |
   | `load(model_dir)` | Load a trained model from a local folder (already downloaded from the run's output). |
   | `convert(model, artifact_id, workdir)` | Write one entry of `artifacts` under `workdir` and return its path. |

5. **Optional hooks**:
   - `available()` returns `None` when ready, or a user-facing reason (a missing optional dependency). Use
     `importlib.util.find_spec`, not an import (see `yolo/__init__.py`).
   - `evaluate(model, df, split_column, item_id_column)` returns an `EvalResult` (report plus predictions). Without
     it the run has no evaluation report. The report's `overall` block feeds the run's headline score; see
     [`scoring/`](../../scoring/README.md).
   - `artifacts` (a dict of `Artifact`) lists what `convert` can produce. The filenames in `ARTIFACT_FILENAMES` are
     the shared convention. An export format is offered for a run only if the run's backend produces the format's
     artifact; see [`export/`](../export/README.md).
   - `metadata_filename` and `preprocessing_manifest(...)` give a bare model export its `preprocessing.json`.
     Without them the export README says the manifest is unavailable.
   - `custom_model_kinds`, `validate_custom_model` and the `custom` argument of `compile` add bring-your-own
     models. Skip them and the backend simply offers none. Set `remote_custom_models = True` only if the
     TRAINED model loads, serves and exports from the run's results alone.
6. **Add the framework dependency** to `server/pyproject.toml`, then check the licence (YOLO is AGPL-3.0).
7. **Add tests**, for example in `server/tests/test_backends.py` style: define the backend, assert it lists, and
   `unregister` it in a `finally` if it is a throwaway.

## Minimal skeleton

```python
# plugins/backends/myfw/__init__.py
import importlib.util
from typing import Any

import pandas as pd

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import (
    ARTIFACT_FILENAMES,
    Artifact,
    CustomModelRef,
    HyperparamsBase,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)


class MyHyperparameters(HyperparamsBase):
    pass  # add extra knobs here


class MyBackend(TrainerBackend):
    id = "myfw"
    label = "My framework"
    description = "One sentence for the backend picker."
    Hyperparameters = MyHyperparameters
    artifacts = {"onnx": Artifact("onnx", ARTIFACT_FILENAMES["onnx"])}

    @classmethod
    def available(cls) -> str | None:
        return None if importlib.util.find_spec("myfw") else "myfw is not installed"

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.id == "image_classification"

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        return [ModelChoice(id="small", label="Small", pretrained=True)]

    @classmethod
    def compile(
        cls, task: TaskDescriptor, ctx: SnapshotContext, hp: MyHyperparameters, custom: CustomModelRef | None = None
    ) -> dict[str, Any]:
        return {"model": hp.model_id or "small", "epochs": hp.epochs or 10}

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        import myfw  # heavy import stays inside the method

        df = pd.read_parquet(run.dataset_uri)
        for epoch in range(run.config["epochs"]):
            run.check_abort()  # raises TrainingAborted if the run was cancelled
            run.heartbeat()
            run.report(epoch, "validation", {"loss": 0.0})
        ...  # write the model under run.output_uri, return it already loaded

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel: ...

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str: ...
```

`LoadedModel` needs `input_columns`, `output` (an `OutputSpec`) and `predict` / `to_output` / `golden_prediction`.
Copy the shape from `yolo/model.py`, the smallest complete example.

## Rules that are easy to break

- **No heavy import at module level.** `list_backends()` imports every backend just to read `id`, `label` and
  `available()`, and the API must start on a machine with no GPU runtime. Import torch, ludwig, ultralytics and so on
  inside the classmethods that need them. Keep a catalog module (like `yolo/tasks.py`) pure data.
- **Class labels come from the trained model**, never from the `label_classes` table. Postgres does not know which
  index a run gave each class, and a wrong order silently mislabels every prediction. `OutputSpec.labels` and any
  export manifest must be read from the model's own artifact (Ludwig's `idx2str`, YOLO's `names.json`).
- **Report a `validation` `loss`** from `train()` when you can. `events/writer.py` uses the lowest validation loss
  to pick a run's best epoch. If you have none, `best_epoch` stays null.
- **`train()` and `evaluate()` may have to run on a remote worker** that has only S3 and an HTTP link to the API
  (`omnitrain/worker/`). Use S3 (`run.dataset_uri`, `run.output_uri`, `services/storage`) and the `TrainContext`
  callbacks. Do not touch Postgres, the event writer, the abort registry, or files on the API host. A framework
  that only reads local files downloads its data into `run.workdir` and uploads results with
  `storage.upload_dir` (see `yolo/data.py` and `yolo/train.py`). If you change what the worker API or a run's S3
  layout looks like, or what `run_core` asks of a backend, bump `PROTOCOL_VERSION` in `schemas/worker.py`.
- **Image resizing**: if your framework does not squash images to a square, say so in the manifest with
  `inputs[].imageResize.mode` (YOLO uses `"shorter_edge_center_crop"`). Generated clients refuse a mode they do
  not know.
- **Layering** is enforced by `tests/test_layering.py`. A backend may import `core/`, `domain/` and
  `services.storage`; do not import `api/` or the job and event machinery.
- **Admin switches** only filter listings and validation of new work. Never filter inside `get_backend`, which
  loads and serves runs that already exist.

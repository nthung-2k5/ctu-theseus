# Export formats

An export format turns a trained run into a downloadable zip: the bare model, a source-only devkit, or an app.
The formats live in [`formats/`](formats/); the rest of this package is the machinery they share (`bundle.py`,
`common.py`, `readme.py`, `metadata.py`, `templates/`).

Adding one needs no enum, migration or frontend edit: `exports.format` is free text holding the plugin id. Restart
the API; `GET /api/export-formats` lists it and the web's single "Export format" combobox renders that list.

## How a format is found

`Registry` imports every module directly under `omnitrain.plugins.export.formats` whose name does not start with
`_`. An `ExportFormat` subclass registers itself when created, but only if it sets its **own** `id`. Shared bases
without an `id` (like `DevkitFormat` and `AppFormat` in `_kits.py`) are helpers, and the underscore keeps that
module from being scanned as a plugin.

## Adding a format

1. **Create a module** `formats/<name>.py`.
2. **Subclass `ExportFormat`** and set:
   - `id`, a stable unique string. It is stored in the database.
   - `label`, and optionally `description` and `notice` (a caveat shown next to the format).
   - `group`, the combobox section: `"Model"`, `"Devkit"` or `"App"` come first in that order; any other string
     sorts alphabetically after them.
   - `order`, the sort key inside a group.
   - `artifact`, the converted-model kind the bundle is built from (`"onnx"`, `"torch_export"`, `"hf_model"`, a
     zipped Hugging Face folder that `assemble` unpacks; see
     `ARTIFACT_FILENAMES` in `plugins/backends/base.py`). A run is only offered formats whose artifact its trainer
     backend actually produces (`Backend.artifacts`).
   - `needs_golden = True` if the bundle embeds a real sample and its expected prediction for verification.
3. **Implement `assemble(ctx)`**, which adds this format's files to the `BundleContext`:
   - `ctx.place_model(prefix="")` writes the model, `preprocessing.json` and `labels.txt`.
   - `ctx.place_golden(prefix="")` writes `expected.json` and the sample (no-op without a golden sample).
   - `ctx.add(path, data)` adds any other file.
   - `cls.readme(ctx)` renders the default README; override `readme` for a custom one.
4. **Optionally override `supports(task)`** to limit the format to some tasks (the default is every task).
5. **Templates**: put client source in `templates/<lang>/` and load it with `template("<lang>/<file>")` from
   `plugins/export/common.py`. `render(...)` does literal `{{VAR}}` substitution, not a template engine.
6. **Test** it in `server/tests/test_export_bundle.py` style: assemble a bundle for a fake `BundleContext` and
   check the files.

## Minimal examples

A bare model, the whole of `formats/onnx_model.py`:

```python
from omnitrain.plugins.export.formats.base import BundleContext, ExportFormat


class OnnxModel(ExportFormat):
    id = "onnx"
    label = "ONNX model"
    description = "The bare .onnx model with preprocessing.json and labels.txt. No client code."
    group = "Model"
    order = 10
    artifact = "onnx"

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        ctx.place_model()
        ctx.add("README.md", cls.readme(ctx))
```

A devkit in a new language reuses `DevkitFormat` and only lists files (see `formats/python_devkit.py`):

```python
from omnitrain.plugins.export.formats._kits import DevkitFormat
from omnitrain.plugins.export.readme import LIMITATION_FULL


class RubyDevkit(DevkitFormat):
    id = "ruby_devkit"
    label = "Ruby devkit"
    description = "ONNX model plus a Ruby inference client."
    order = 60
    readme_info = {
        "label": "Ruby",
        "files": "`omnitrain_client.rb`, `example.rb`",
        "dependencies": "gem install onnxruntime",
        "run_cmd": "ruby example.rb <path-to-image>",
        "verify_cmd": "ruby verify.rb",
        "limitation": LIMITATION_FULL,
    }
    client_files = (("omnitrain_client.rb", "ruby/omnitrain_client.rb"), ("example.rb", "ruby/example.rb"))
    verify_files = (("verify.rb", "ruby/verify.rb"),)
```

The `ruby/` templates go under `templates/ruby/`.

## Rules that are easy to break

- **Class order comes from the trained model**, carried in `ctx.preprocessing` and `ctx.labels`, never from the
  `label_classes` table. Never re-derive it in a format.
- **A client only runs the kinds it implements.** `client_kind(task)` in `_kits.py` says whether a task is
  `standard` (classification, regression), `detection` (NMS and an undone letterbox) or `video` (decode, sample
  clips); `DevkitFormat.handles` lists the kinds a devkit's client really implements, and `supports` offers it for
  those only. Today every devkit and app handles `standard` and `detection`; only the Python devkit also handles `video`. Add a kind
  to `handles` only together with the client code, its `kinds` README text and a verify branch.
- **Looking a format up by id is not filtered.** `find_export_format` and `get_export_format` must keep working
  after an admin switches a format off, so an export that was already queued can finish. Admin switches apply
  only to `list_export_formats` and to validating new work.
- **The expected-prediction shape is a contract.** Every generated client parses `expected.json`; do not change
  `LoadedModel.golden_prediction`'s shape for a single format.
- **Conversion is not your job.** `assemble` lays out files from `ctx.model_bytes`. Turning the trained model into
  ONNX or `.pt2` is `TrainerBackend.convert`.

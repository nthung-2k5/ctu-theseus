"""README generation for export bundles (ported from server/lib/export/readme.ts).

Plain string building rather than a file template: the content genuinely branches on the kind of
bundle, and this is interpolation, not a templating language. Per-format facts (dependencies, run
commands) live on the export format classes and are passed in as `info`.
"""

from dataclasses import dataclass
from typing import Any

LIMITATION_FULL = (
    "Implements preprocessing for **image classification/regression** and **tabular (numeric-only) inputs**. "
    "For other input types (text, audio), `preprocessing.json` still has the exact parameters Ludwig used "
    "(`inputs[0].ludwigPreprocessing`); the client raises a clear error rather than silently producing wrong "
    "predictions. Adapt the client's preprocessing method for your input type."
)
LIMITATION_IMAGE_ONLY = (
    "Only implements preprocessing for **image classification** inputs today. For other input types, "
    "`preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`); the "
    "client raises a clear error rather than silently producing wrong predictions. Adapt the client's "
    "preprocessing method for your input type."
)


@dataclass
class ReadmeVars:
    run_name: str
    task_label: str
    format: str  # the export format's display label
    artifact: str  # model file name inside the bundle, e.g. model.onnx
    has_verify: bool


def _header(v: ReadmeVars) -> str:
    return f"# {v.run_name}\n\nExported from Theseus: task **{v.task_label}**, format **{v.format}**.\n"


def model_readme(v: ReadmeVars) -> str:
    return f"""{_header(v)}
## Contents

- `{v.artifact}`: the trained model artifact.
- `preprocessing.json`: the exact preprocessing Ludwig applied at training time (image resize/normalize, tokenizer params, ...) and, for classification outputs, the class list in the model's internal index order.
- `labels.txt`: one class name per line, in the same order as `preprocessing.json`'s output classes (if this is a classification task).

This is the bare artifact only, with no client code. `preprocessing.json` has everything needed to reconstruct the input pipeline yourself; see the devkit export formats for a working reference implementation.
"""


def app_readme(v: ReadmeVars, info: dict[str, Any]) -> str:
    verify = ""
    if v.has_verify:
        verify = (
            "\n## Verify\n\nThis bundle includes one real test-split sample and the platform's own prediction "
            f"for it (`expected.json`). {info['verify_note']} It confirms this app's own preprocessing "
            "reproduces that prediction within tolerance: the only proof in this bundle that its "
            "re-implementation of Ludwig's preprocessing is correct.\n"
        )
    return f"""{_header(v)}
A {info["label"]}: runs fully on-device (or in-browser), no server involved.

## Limitation

Only implements preprocessing for **image classification** inputs today. `preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`) for other input types; adapt the client code for your input type.

## Quick start

```bash
{info["setup_cmd"]}
```

{info["run_cmd"]}
{verify}"""


def devkit_readme(v: ReadmeVars, info: dict[str, Any]) -> str:
    contents_extra = ", `expected.json`, `sample/`" if v.has_verify else ""
    note = f"\n{info['note']}\n" if info.get("note") else ""
    verify = ""
    if v.has_verify:
        verify = (
            "\n## Verify\n\nThis bundle includes one real test-split sample and the platform's own prediction "
            f"for it (`expected.json`). Run:\n\n```bash\n{info['verify_cmd']}\n```\n\n"
            "to confirm this client's preprocessing reproduces that prediction within tolerance.\n"
        )
    return f"""{_header(v)}
Generated {info["label"]} inference client: wraps the ONNX model with the same preprocessing/postprocessing Ludwig used at training time (see `preprocessing.json`). Ships as **source only**, with no project or build file, so you can drop it straight into an existing project.

## Contents

{info["files"]}, `preprocessing.json`, `labels.txt`{contents_extra}.

## Limitation

{info["limitation"]}

## Dependencies

```
{info["dependencies"]}
```
{note}
## Quick start

```bash
{info["run_cmd"]}
```
{verify}"""

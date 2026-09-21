"""README generation for export bundles (ported from server/lib/export/readme.ts).

Plain string building rather than a file template: the content genuinely branches on tier and
language, and this is interpolation, not a templating language.
"""

from dataclasses import dataclass
from typing import Any

_LIMITATION_FULL = (
    "Implements preprocessing for **image classification/regression** and **tabular (numeric-only) inputs**. "
    "For other input types (text, audio), `preprocessing.json` still has the exact parameters Ludwig used "
    "(`inputs[0].ludwigPreprocessing`); the client raises a clear error rather than silently producing wrong "
    "predictions. Adapt the client's preprocessing method for your input type."
)
_LIMITATION_IMAGE_ONLY = (
    "Only implements preprocessing for **image classification** inputs today. For other input types, "
    "`preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`); the "
    "client raises a clear error rather than silently producing wrong predictions. Adapt the client's "
    "preprocessing method for your input type."
)

# Devkit languages ship source only (no project or build files), so dependencies and run commands live here.
DEVKIT_INFO: dict[str, dict[str, Any]] = {
    "python": {
        "label": "Python",
        "files": "`theseus_client.py`, `example.py`",
        "dependencies": "pip install onnxruntime numpy pillow",
        "run_cmd": "python example.py <path-to-image>\npython example.py '{\"column\": value, ...}'   # tabular models",
        "verify_cmd": "python verify.py",
        "limitation": _LIMITATION_FULL,
    },
    "typescript": {
        "label": "TypeScript (Bun)",
        "files": "`client.ts`, `example.ts`",
        "dependencies": "bun add onnxruntime-node sharp",
        "run_cmd": "bun example.ts <path-to-image>\nbun example.ts '{\"column\": value, ...}'   # tabular models",
        "verify_cmd": "bun verify.ts",
        "limitation": _LIMITATION_FULL,
    },
    "csharp": {
        "label": "C#",
        "files": "`TheseusClient.cs`, `Program.cs`",
        "dependencies": (
            "dotnet add package Microsoft.ML.OnnxRuntime --version 1.19.2\n"
            "dotnet add package SixLabors.ImageSharp --version 3.1.12"
        ),
        "run_cmd": "dotnet run -- <path-to-image>",
        "verify_cmd": "dotnet run -- verify",
        "limitation": _LIMITATION_IMAGE_ONLY,
        "note": (
            "`Program.cs` uses top-level statements as its entry point. If your project already has one, "
            "merge its logic in rather than copying the file verbatim."
        ),
    },
    "java": {
        "label": "Java / Kotlin",
        "files": "`TheseusClient.java`, `Main.java`",
        "dependencies": (
            "// Gradle (Kotlin DSL)\n"
            'implementation("com.microsoft.onnxruntime:onnxruntime:1.29.0")\n'
            'implementation("com.fasterxml.jackson.core:jackson-databind:2.22.1")'
        ),
        "run_cmd": "java Main <path-to-image>",
        "verify_cmd": "java Main verify",
        "limitation": _LIMITATION_IMAGE_ONLY,
        "note": "Written in Java for maximum interop: call it directly from Kotlin with zero wrapping.",
    },
}

APP_INFO: dict[str, dict[str, str]] = {
    "pwa": {
        "label": "Progressive Web App",
        "setup_cmd": "python -m http.server 8000   # or: npx serve .",
        "run_cmd": 'Open http://localhost:8000 in a browser, then use the browser\'s "Install" prompt.',
        "verify_note": 'Use the "Run built-in self-check" button on the page.',
    },
    "flutter": {
        "label": "Flutter app",
        "setup_cmd": "flutter pub get",
        "run_cmd": "flutter run   # or: flutter build apk / flutter build ios",
        "verify_note": "Tap the checkmark icon in the app bar to run the built-in self-check screen.",
    },
}


@dataclass
class ReadmeVars:
    run_name: str
    task_label: str
    format: str
    tier: str
    has_verify: bool


def render_readme(v: ReadmeVars, lang: str | None) -> str:
    header = f"# {v.run_name}\n\nExported from Theseus: task **{v.task_label}**, format **{v.format}**.\n"

    if v.tier == "model":
        return f"""{header}
## Contents

- `model.{v.format}`: the trained model artifact.
- `preprocessing.json`: the exact preprocessing Ludwig applied at training time (image resize/normalize, tokenizer params, ...) and, for classification outputs, the class list in the model's internal index order.
- `labels.txt`: one class name per line, in the same order as `preprocessing.json`'s output classes (if this is a classification task).

This is the bare artifact only, with no client code. `preprocessing.json` has everything needed to reconstruct the input pipeline yourself; see the `devkit` export tier for a working reference implementation.
"""

    if v.tier == "app":
        info = APP_INFO[lang or ""]
        verify = ""
        if v.has_verify:
            verify = (
                "\n## Verify\n\nThis bundle includes one real test-split sample and the platform's own prediction "
                f"for it (`expected.json`). {info['verify_note']} It confirms this app's own preprocessing "
                "reproduces that prediction within tolerance: the only proof in this bundle that its "
                "re-implementation of Ludwig's preprocessing is correct.\n"
            )
        return f"""{header}
A {info["label"]}: runs fully on-device (or in-browser), no server involved.

## Limitation

Only implements preprocessing for **image classification** inputs today. `preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`) for other input types; adapt the client code for your input type.

## Quick start

```bash
{info["setup_cmd"]}
```

{info["run_cmd"]}
{verify}"""

    # devkit
    info = DEVKIT_INFO[lang or ""]
    contents_extra = ", `expected.json`, `sample/`" if v.has_verify else ""
    note = f"\n{info['note']}\n" if info.get("note") else ""
    verify = ""
    if v.has_verify:
        verify = (
            "\n## Verify\n\nThis bundle includes one real test-split sample and the platform's own prediction "
            f"for it (`expected.json`). Run:\n\n```bash\n{info['verify_cmd']}\n```\n\n"
            "to confirm this client's preprocessing reproduces that prediction within tolerance.\n"
        )
    return f"""{header}
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

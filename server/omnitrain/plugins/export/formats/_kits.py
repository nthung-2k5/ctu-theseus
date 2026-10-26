"""Shared bases for source-only devkits and installable apps. No `id`, so these never register."""

from typing import Any, ClassVar

from omnitrain.domain.task_registry import TaskDescriptor
from omnitrain.plugins.export.common import template
from omnitrain.plugins.export.formats.base import BundleContext, ExportFormat
from omnitrain.plugins.export.readme import app_readme, devkit_readme

STANDARD, DETECTION, VIDEO, AUDIO = "standard", "detection", "video", "audio"


def client_kind(task: TaskDescriptor) -> str:
    """What a generated client has to do for this task: `standard` (classification and regression: one preprocessed
    input, a flat `{label: value}` result), `detection` (the graph returns every candidate box, so the client runs
    the confidence filter, non-maximum suppression and undoes the letterbox), `video` (decode a file, sample clips,
    average the clips' probabilities) or `audio` (decode a recording, resample it, cut windows, make a log-mel
    spectrogram of each and average the windows' probabilities)."""
    if task.output is not None and task.output.kind == "detections":
        return DETECTION
    if task.modality == "video":
        return VIDEO
    if task.modality == "audio":
        return AUDIO
    return STANDARD


def supports_generated_clients(task: TaskDescriptor) -> bool:
    """Whether every generated client (not only the ones that learned a kind) can run the task's model."""
    return client_kind(task) == STANDARD


class DevkitFormat(ExportFormat):
    """A generated inference client shipped as source only, wrapping the ONNX model.

    Subclasses list (bundle path, template path) pairs; `verify_files` are only added when the
    bundle carries a golden sample to verify against.
    """

    group = "Devkit"
    notice = (
        "Devkit bundles ship source only (no project or build files; dependencies are documented in the README). "
        "Preprocessing is implemented for image and tabular inputs, object detection and audio (a trained "
        "PyTorch model); video is offered by the Python devkit only. See the bundle's README for other modalities."
    )
    artifact = "onnx"
    needs_golden = True

    readme_info: ClassVar[dict[str, Any]]
    client_files: ClassVar[tuple[tuple[str, str], ...]]
    verify_files: ClassVar[tuple[tuple[str, str], ...]] = ()
    # The `client_kind`s this client implements; a kind it lacks is exported as the bare model only.
    handles: ClassVar[frozenset[str]] = frozenset({STANDARD})

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return client_kind(task) in cls.handles

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        ctx.place_model()
        ctx.place_golden()
        for path, tpl in cls.client_files:
            ctx.add(path, template(tpl))
        if ctx.golden:
            for path, tpl in cls.verify_files:
                ctx.add(path, template(tpl))
        ctx.add("README.md", cls.readme(ctx))

    @classmethod
    def readme(cls, ctx: BundleContext) -> str:
        return devkit_readme(ctx.readme_vars, cls.readme_info)


class AppFormat(ExportFormat):
    """A runnable on-device app around the ONNX model."""

    group = "App"
    notice = (
        "App bundles run inference on-device, never on a server. Preprocessing is implemented for image "
        "classification, object detection and audio classification. See the bundle's README for other modalities."
    )
    artifact = "onnx"
    needs_golden = True

    readme_info: ClassVar[dict[str, Any]]
    # The `client_kind`s this app implements; a kind it lacks is exported as the bare model only.
    handles: ClassVar[frozenset[str]] = frozenset({STANDARD})

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return client_kind(task) in cls.handles

    @classmethod
    def readme(cls, ctx: BundleContext) -> str:
        return app_readme(ctx.readme_vars, cls.readme_info)

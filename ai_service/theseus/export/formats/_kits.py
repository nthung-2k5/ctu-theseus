"""Shared bases for source-only devkits and installable apps. No `id`, so these never register."""

from typing import Any, ClassVar

from theseus.export.common import template
from theseus.export.formats.base import BundleContext, ExportFormat
from theseus.export.readme import app_readme, devkit_readme


class DevkitFormat(ExportFormat):
    """A generated inference client shipped as source only, wrapping the ONNX model.

    Subclasses list (bundle path, template path) pairs; `verify_files` are only added when the
    bundle carries a golden sample to verify against.
    """

    group = "Devkit"
    notice = (
        "Devkit bundles ship source only (no project or build files; dependencies are documented in the README). "
        "Preprocessing is currently implemented for image classification only. See the bundle's README for other "
        "modalities."
    )
    artifact = "onnx"
    needs_golden = True

    readme_info: ClassVar[dict[str, Any]]
    client_files: ClassVar[tuple[tuple[str, str], ...]]
    verify_files: ClassVar[tuple[tuple[str, str], ...]] = ()

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
        "App bundles run inference on-device, never on a server. Preprocessing is currently implemented for image "
        "classification only. See the bundle's README for other modalities."
    )
    artifact = "onnx"
    needs_golden = True

    readme_info: ClassVar[dict[str, Any]]

    @classmethod
    def readme(cls, ctx: BundleContext) -> str:
        return app_readme(ctx.readme_vars, cls.readme_info)

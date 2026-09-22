"""The export format plugin base class.

To add a format: create a module in this package with a subclass that sets `id`, `label` and
`artifact`, and implement `assemble`. Restart the backend; GET /api/export-formats returns it and
the web renders it, with no enum, migration or frontend change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from theseus.export.common import BundleFile, GoldenSample, dumps
from theseus.export.readme import ReadmeVars, model_readme
from theseus.plugins import Registry
from theseus.services.task_registry import TaskDescriptor

_registry: Registry["ExportFormat"] = Registry("export format", "theseus.export.formats")


@dataclass
class BundleContext:
    """Everything a format needs to lay out its zip, plus the helpers that write into it."""

    run_name: str
    task_label: str
    format_label: str
    # File name of the model artifact inside the bundle, e.g. model.onnx.
    artifact_filename: str
    model_bytes: bytes
    preprocessing: dict[str, Any]
    golden: GoldenSample | None
    files: list[BundleFile] = field(default_factory=list)

    @property
    def labels(self) -> list[str] | None:
        # Class order comes from Ludwig's idx2str, carried in preprocessing.json, never from Postgres.
        return next((o["classes"] for o in self.preprocessing["outputs"] if o.get("classes")), None)

    @property
    def readme_vars(self) -> ReadmeVars:
        return ReadmeVars(
            self.run_name, self.task_label, self.format_label, self.artifact_filename, self.golden is not None
        )

    def add(self, path: str, data: str | bytes, compress: bool = True) -> None:
        self.files.append(BundleFile(path, data.encode() if isinstance(data, str) else data, compress))

    def place_model(self, prefix: str = "") -> None:
        self.add(f"{prefix}{self.artifact_filename}", self.model_bytes, compress=False)
        self.add(f"{prefix}preprocessing.json", dumps(self.preprocessing))
        if self.labels:
            self.add(f"{prefix}labels.txt", "\n".join(self.labels))

    def place_golden(self, prefix: str = "") -> None:
        if self.golden is None:
            return
        self.add(f"{prefix}expected.json", self.golden.expected_json)
        self.add(f"{prefix}sample/{self.golden.sample_filename}", self.golden.sample_bytes)


class ExportFormat(ABC):
    id: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str] = ""
    # Optional caveat shown next to the format in the UI (limitations, requirements). Empty for none.
    notice: ClassVar[str] = ""
    # Section heading in the web combobox ("Model", "Devkit", "App"); free text, new groups are fine.
    group: ClassVar[str] = "Model"
    # Sort key inside a group.
    order: ClassVar[int] = 100
    # Key in export.artifacts.ARTIFACTS: which converted model this format is built from.
    artifact: ClassVar[str] = "onnx"
    # Whether the bundle embeds a real test sample and its prediction (a golden sample) for verification.
    needs_golden: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _registry.register(cls)

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        """Whether this format can be offered for a project's task."""
        return True

    @classmethod
    @abstractmethod
    def assemble(cls, ctx: BundleContext) -> None:
        """Add this format's files to `ctx` (ctx.add / place_model / place_golden)."""

    @classmethod
    def readme(cls, ctx: BundleContext) -> str:
        return model_readme(ctx.readme_vars)

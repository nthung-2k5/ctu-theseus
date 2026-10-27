"""The export format plugin base class.

To add a format: create a module in this package with a subclass that sets `id`, `label` and
`artifact`, and implement `assemble`. Restart the backend; GET /api/export-formats returns it and
the web renders it, with no enum, migration or frontend change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from omnitrain.core.plugins import Registry
from omnitrain.domain.task_registry import TaskDescriptor
from omnitrain.plugins.export.common import BundleFile, GoldenSample, dumps
from omnitrain.plugins.export.readme import ReadmeVars, model_readme

_registry: Registry["ExportFormat"] = Registry("export format", "omnitrain.plugins.export.formats")

TOKENIZER_FILENAME = "tokenizer.json"


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
        # Class order comes from the trained model itself, carried in preprocessing.json, never from Postgres.
        return next((o["classes"] for o in self.preprocessing["outputs"] if o.get("classes")), None)

    @property
    def readme_vars(self) -> ReadmeVars:
        return ReadmeVars(
            self.run_name,
            self.task_label,
            self.format_label,
            self.artifact_filename,
            self.golden is not None,
            self.kind,
        )

    @property
    def kind(self) -> str:
        """Which client kind the manifest describes: detection, video, audio, text, hf_text, tags, hf_tags or
        standard."""
        if any(o.get("type") == "detections" for o in self.preprocessing.get("outputs", [])):
            return "detection"
        if any(o.get("type") == "tags" for o in self.preprocessing.get("outputs", [])):
            # A tagger reads a word per position, or, when it is a Hugging Face model, whatever pieces its own
            # tokenizer cuts the words into.
            has_tokenizer = any("hfTokenizer" in i for i in self.preprocessing.get("inputs", []))
            return "hf_tags" if has_tokenizer else "tags"
        if any(i.get("type") == "video" for i in self.preprocessing.get("inputs", [])):
            return "video"
        if any(i.get("type") == "audio" and "audioSpectrogram" in i for i in self.preprocessing.get("inputs", [])):
            return "audio"
        if any(i.get("type") == "text" and "textTokens" in i for i in self.preprocessing.get("inputs", [])):
            return "text"
        if any(i.get("type") == "text" and "hfTokenizer" in i for i in self.preprocessing.get("inputs", [])):
            return "hf_text"
        return "standard"

    def add(self, path: str, data: str | bytes, compress: bool = True) -> None:
        self.files.append(BundleFile(path, data.encode() if isinstance(data, str) else data, compress))

    def place_model(self, prefix: str = "") -> None:
        self.add(f"{prefix}{self.artifact_filename}", self.model_bytes, compress=False)
        self.add(f"{prefix}preprocessing.json", dumps(self._manifest_without_tokenizer(prefix)))
        if self.labels:
            self.add(f"{prefix}labels.txt", "\n".join(self.labels))

    def _manifest_without_tokenizer(self, prefix: str) -> dict[str, Any]:
        """A Hugging Face model's tokenizer is a JSON document of its own, megabytes for a multilingual vocabulary, and
        the file every Hugging Face library reads. It goes beside the manifest as `tokenizer.json`, and the manifest
        says so (`tokenizerFile`) instead of carrying it a second time."""
        inputs = []
        for spec in self.preprocessing.get("inputs", []):
            recipe = spec.get("hfTokenizer")
            if isinstance(recipe, dict) and isinstance(recipe.get("tokenizer"), str):
                self.add(f"{prefix}{TOKENIZER_FILENAME}", recipe["tokenizer"])
                recipe = {**{k: v for k, v in recipe.items() if k != "tokenizer"}, "tokenizerFile": TOKENIZER_FILENAME}
                spec = {**spec, "hfTokenizer": recipe}
            inputs.append(spec)
        return {**self.preprocessing, "inputs": inputs}

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

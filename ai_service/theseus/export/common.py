"""Small pure helpers shared by bundle assembly and the export format classes."""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TEMPLATES = Path(__file__).parent / "templates"


@dataclass
class BundleFile:
    path: str
    data: bytes
    # The model artifact is already compact binary: store it (level 0) instead of deflating.
    compress: bool = True


@dataclass
class GoldenSample:
    expected_json: bytes
    sample_filename: str
    sample_bytes: bytes


@lru_cache
def template(relative: str) -> str:
    return (TEMPLATES / relative).read_text(encoding="utf-8")


def render(tpl: str, variables: dict[str, str]) -> str:
    """{{VAR}} substitution for README and manifest-ish files: deliberately a literal replace, not a template engine."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: variables.get(m.group(1), ""), tpl)


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def dart_package_name(name: str) -> str:
    """A valid pubspec `name:`: lowercase_with_underscores, starting with a letter."""
    snake = re.sub(r"^_+|_+$", "", re.sub(r"[^a-z0-9]+", "_", name.lower()))
    if not snake:
        return "theseus_app"
    return snake if re.match(r"[a-z]", snake) else f"app_{snake}"

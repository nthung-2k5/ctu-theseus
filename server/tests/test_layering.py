"""The import layering of `omnitrain/` (see "Layering" in CLAUDE.md), checked from the source.

Plain `ast`, no extra dependency. Imports inside a function count too: a lazy import is still a dependency.

A target is a top-level package/module (`core`, `domain`, `settings`), except under `services`, `jobs` and `db`, where
it is the submodule (`services.storage`, `db.enums`): those three packages are mixed, so the rule has to say which part
a layer may use. `plugins/` and `api/` are only namespaces: `plugins/backends` is the layer `backends`, and
`api/routers` is the layer `routers`. Every exception below is deliberate and has a reason; a new violation fails
here instead of landing.
"""

import ast
from collections import defaultdict
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "omnitrain"
SPLIT_PACKAGES = {"services", "jobs", "db"}
NAMESPACES = {"plugins", "api"}
# The HTTP layer: nothing outside it may import it.
HTTP = {"app", "deps", "lifespan", "errors", "routers"}


def _targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
        mods = [n.split(".") for n in names if n == "omnitrain" or n.startswith("omnitrain.")]
        return [_target(parts[1:], None) for parts in mods if len(parts) > 1]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        parts = node.module.split(".")
        if parts[0] != "omnitrain":
            return []
        if len(parts) == 1:  # from omnitrain import settings / constants
            return [_target([a.name], None) for a in node.names]
        return [_target(parts[1:], [a.name for a in node.names])]
    return []


def _target(parts: list[str], names: list[str] | None) -> str:
    if parts[0] in NAMESPACES:
        if len(parts) > 1:
            parts = parts[1:]
        elif names:  # `from omnitrain.api import deps`
            parts = [names[0]]
    head = parts[0]
    if head in SPLIT_PACKAGES:
        if len(parts) > 1:
            return f"{head}.{parts[1]}"
        if names:  # `from omnitrain.services import storage`: the imported name is the submodule
            return f"{head}.{names[0]}"
    return head


def _imports() -> dict[str, dict[str, list[str]]]:
    """layer -> target -> ['path:line', ...]"""
    found: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG)
        if rel.parts[0] in NAMESPACES:
            layer = rel.parts[1] if len(rel.parts) > 2 else rel.stem
        else:
            layer = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        if layer == "__init__":
            continue
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            for target in _targets(node):
                if target.split(".")[0] != layer:
                    found[layer][target].append(f"{rel.as_posix()}:{node.lineno}")
    return found


# Layers that may import ONLY what is listed. (A layer's own modules are always fine.)
ALLOW: dict[str, set[str]] = {
    "core": set(),
    "scoring": {"core"},
    "sweep_strategies": {"core"},
    "tasks": {"core", "db.enums", "schemas"},
    "domain": {
        "core",
        "db.enums",
        "schemas",
        "tasks",  # task_registry is the lookup facade over the task plugins
        "backends",  # classification_report reads a backend's EvalResult/LoadedModel types
    },
    "schemas": {"db.enums", "domain", "augmentation", "preprocessing"},
    "db": {"settings"},
    "auth": {"db.models", "settings"},
    "events": {"core", "db.base", "db.enums", "db.models"},
    # Plugins: they reach for core/domain. Exceptions: S3 (a backend's train/evaluate may use it by contract) and the
    # admin on/off switch every plugin registry applies to its listings.
    "backends": {"core", "domain", "schemas", "settings", "services.storage", "services.plugin_settings"},
    "augmentation": {"core", "domain", "db.enums", "schemas", "preprocessing", "services.plugin_settings"},
    "preprocessing": {"core", "domain", "db.enums", "schemas", "augmentation", "services.plugin_settings"},
    "export": {
        "core",
        "domain",
        "backends",
        "services.storage",
        "services.plugin_settings",
        # Known, not wanted: a format reads the run and its model row to assemble a bundle.
        "db.base",
        "db.models",
    },
    "model_sources": {
        "core",
        "db.models",
        # A source is tied to the custom-model machinery it fetches for.
        "services.custom_models",
        "services.integrations",
    },
}

# Layers that may import anything EXCEPT what is listed.
DENY: dict[str, set[str]] = {
    # services -> jobs is the enqueue-and-wake direction (the reverse, jobs -> services, is how the work runs), so a
    # service may only use the abort registry and the dispatcher's nudge.
    "services": {"routers", "worker"},
    "jobs": {"routers", "worker"},
    "worker": {"routers"},
}
SERVICES_MAY_USE_JOBS = {"jobs.abort", "jobs.dispatcher"}


@pytest.fixture(scope="module")
def imports():
    return _imports()


@pytest.mark.parametrize("layer", sorted(ALLOW))
def test_strict_layers_import_only_what_they_may(imports, layer):
    bad = {t: where for t, where in imports[layer].items() if t not in ALLOW[layer]}
    assert not bad, f"`{layer}` may import only {sorted(ALLOW[layer])}, but imports:\n" + "\n".join(
        f"  {t}  <- {', '.join(where[:3])}" for t, where in sorted(bad.items())
    )


@pytest.mark.parametrize("layer", sorted(DENY))
def test_loose_layers_do_not_import_upward(imports, layer):
    bad = {t: where for t, where in imports[layer].items() if t in DENY[layer]}
    assert not bad, f"`{layer}` must not import {sorted(DENY[layer])}:\n" + "\n".join(
        f"  {t}  <- {', '.join(where[:3])}" for t, where in sorted(bad.items())
    )


def test_services_use_only_the_abort_registry_and_the_dispatcher_from_jobs(imports):
    jobs = {t for t in imports["services"] if t.startswith("jobs")}
    assert jobs <= SERVICES_MAY_USE_JOBS, f"services may use {sorted(SERVICES_MAY_USE_JOBS)} from jobs, not {jobs}"


def test_only_the_http_layer_imports_the_http_layer(imports):
    offenders = [
        (layer, where[0])
        for layer, targets in imports.items()
        for target, where in targets.items()
        if target in HTTP and layer not in HTTP
    ]
    assert not offenders, f"only app/deps/lifespan/errors/routers may import each other, not: {offenders}"

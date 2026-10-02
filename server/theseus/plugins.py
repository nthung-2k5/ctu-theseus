"""Class-based plugin registries: define a subclass, restart the backend, it shows up.

A registry owns one base class (export formats, augmentations) and one package. Subclasses that
set their own `id` register themselves at class-creation time (`__init_subclass__`); intermediate
base classes that do not define `id` are helpers and stay out of the registry. The package is
imported lazily, once, on first lookup, so dropping a module into it is all a plugin author does.
Registries are immutable after startup: there is no hot reload, a restart is the reload story.
"""

import importlib
import pkgutil
import threading
from typing import Any


class Registry[T]:
    def __init__(self, kind: str, package: str) -> None:
        self.kind = kind
        self.package = package
        self._items: dict[str, type[T]] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def register(self, cls: type[T]) -> None:
        # Only classes that declare `id` themselves count; a subclass of a concrete plugin
        # must not silently inherit (and collide on) its parent's id.
        plugin_id = cls.__dict__.get("id")
        if plugin_id is None:
            return
        existing = self._items.get(plugin_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"Duplicate {self.kind} id {plugin_id!r}: {existing.__module__} and {cls.__module__}")
        self._items[plugin_id] = cls

    def unregister(self, plugin_id: str) -> None:
        """For tests that define throwaway plugins."""
        self._items.pop(plugin_id, None)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            load_plugins(self.package)
            self._loaded = True

    def get(self, plugin_id: str) -> type[T]:
        self.ensure_loaded()
        try:
            return self._items[plugin_id]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} {plugin_id!r}") from None

    def find(self, plugin_id: str) -> type[T] | None:
        self.ensure_loaded()
        return self._items.get(plugin_id)

    def all(self) -> list[type[T]]:
        self.ensure_loaded()
        return list(self._items.values())


def load_plugins(package_name: str) -> list[Any]:
    """Import every module directly under `package_name` so its plugin classes register."""
    package = importlib.import_module(package_name)
    return [
        importlib.import_module(f"{package_name}.{info.name}")
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_")
    ]

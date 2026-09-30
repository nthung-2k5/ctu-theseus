"""Workarounds for Ludwig bugs that only bite on Windows (development machines; the service itself runs on Linux).

Importing this module applies them. It imports `ludwig`, so like `train.py` it is only imported lazily.
"""

import errno
import functools
import re
import sys
from collections.abc import Callable
from typing import Any

import fsspec
from ludwig.models.ecd import ECD
from ludwig.utils import fs_utils

_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _tolerate_fsync_on_read_only_handle(save):
    """`ECD.save` writes the safetensors weights, then reopens the file `"rb"` just to `os.fsync` it.
    Windows only lets you fsync a handle opened for writing, so that raises `OSError(EBADF)` ("Bad file
    descriptor") and fails the whole training run after the weights are already on disk. That fsync is the
    last thing `save` does, so swallowing EBADF there loses nothing; any other error still propagates."""

    @functools.wraps(save)
    def wrapper(self, save_path):
        try:
            return save(self, save_path)
        except OSError as e:
            if e.errno != errno.EBADF:
                raise

    wrapper._theseus_patched = True  # type: ignore[attr-defined]
    return wrapper


def _keep_windows_drive_letter(get_fs_and_path: Callable[[str], tuple[Any, str]]):
    """`fs_utils.get_fs_and_path` runs `urlparse` on the path, which reads `C:\\Users\\x` as scheme `C` and keeps
    only `\\Users\\x`. Every local directory Ludwig creates (`makedirs`, `path_exists`, ...) then resolves against
    the CURRENT drive: train with a `C:` output directory from an `E:` working directory and the model is written
    to `E:\\Users\\...`, while the returned path points at an empty `C:` one. Hand a drive-letter path straight to
    the local filesystem instead; everything else (URLs, `file://`, POSIX paths) still goes through Ludwig."""

    @functools.wraps(get_fs_and_path)
    def wrapper(url):
        if isinstance(url, str) and _WINDOWS_DRIVE_PATH.match(url):
            return fsspec.filesystem("file"), url
        return get_fs_and_path(url)

    wrapper._theseus_patched = True  # type: ignore[attr-defined]
    return wrapper


if sys.platform == "win32":
    if not getattr(ECD.save, "_theseus_patched", False):
        ECD.save = _tolerate_fsync_on_read_only_handle(ECD.save)  # type: ignore[method-assign]
    if not getattr(fs_utils.get_fs_and_path, "_theseus_patched", False):
        fs_utils.get_fs_and_path = _keep_windows_drive_letter(fs_utils.get_fs_and_path)

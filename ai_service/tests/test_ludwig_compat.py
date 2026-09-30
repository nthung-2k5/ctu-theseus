"""Windows-only Ludwig workarounds: `ECD.save` fsyncs a read-only handle (EBADF), and `get_fs_and_path` drops the
drive letter from an absolute path."""

import os
import sys

import pytest

pytest.importorskip("ludwig")


def test_training_saves_the_model_under_an_absolute_output_directory(tmp_path):
    """Both Windows bugs at once: the save must not raise (fsync), and it must land under `tmp_path` even when
    that is on another drive than the cwd (drive letter)."""
    import numpy as np
    import pandas as pd
    from ludwig.api import LudwigModel

    from theseus.backends.ludwig import compat  # noqa: F401

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.random(60)})
    df["y"] = np.where(df.a > 0.5, "hi", "lo")
    config = {
        "input_features": [{"name": "a", "type": "number"}],
        "output_features": [{"name": "y", "type": "category"}],
        "trainer": {"epochs": 1, "batch_size": 16},
        "backend": {"type": "local"},
    }
    _, _, output_dir = LudwigModel(config, logging_level=40).train(
        dataset=df, output_directory=str(tmp_path), skip_save_processed_input=True
    )

    assert os.path.isdir(output_dir)  # the returned directory is the one that was written
    assert os.path.commonpath([os.path.abspath(output_dir), str(tmp_path)]) == str(tmp_path)
    weights = [f for _, _, files in os.walk(output_dir) for f in files if f.endswith(".safetensors")]
    assert weights  # the weights really were written, not just the error swallowed


def test_drive_letter_paths_go_straight_to_the_local_filesystem():
    from theseus.backends.ludwig.compat import _keep_windows_drive_letter

    calls = []

    def ludwig_original(url):
        calls.append(url)
        return "ludwig-fs", "ludwig-path"

    wrapped = _keep_windows_drive_letter(ludwig_original)

    for url in ("C:\\Users\\x\\out", "D:/data/out"):
        fs, path = wrapped(url)
        assert type(fs).__name__ == "LocalFileSystem" and path == url
    assert calls == []  # never handed to urlparse, which would read `C` as a URL scheme

    for url in ("/tmp/out", "relative/out", "s3://bucket/key", "file:///tmp/out"):
        assert wrapped(url) == ("ludwig-fs", "ludwig-path")
    assert calls == ["/tmp/out", "relative/out", "s3://bucket/key", "file:///tmp/out"]


@pytest.mark.skipif(sys.platform != "win32", reason="the patch is only applied on Windows")
def test_ludwig_keeps_the_drive_letter_once_patched():
    from ludwig.utils import fs_utils

    from theseus.backends.ludwig import compat  # noqa: F401

    _, path = fs_utils.get_fs_and_path("C:\\Users\\x\\out")
    assert path.replace("\\", "/").lower().startswith("c:/users/x/out")


def test_other_oserrors_still_propagate():
    import errno

    from theseus.backends.ludwig.compat import _tolerate_fsync_on_read_only_handle

    def boom(self, path):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(OSError, match="No space"):
        _tolerate_fsync_on_read_only_handle(boom)(None, "x")

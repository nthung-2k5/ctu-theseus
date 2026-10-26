"""The generated Python devkit client for an audio classifier.

The model is built by hand (a pooled linear map over the spectrogram), so what is under test is the client's own
arithmetic: WAV decoding, resampling, windowing and the log-mel spectrogram, each against the backend's reference in
`plugins/backends/pytorch/audio.py` (and scipy for the resampler), which is what the manifest's `audioSpectrogram`
describes.
"""

import json
import struct
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from scipy.io import wavfile
from scipy.signal import resample_poly

from omnitrain.plugins.backends.pytorch import audio
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.pytorch.spec import AudioSpec

TEMPLATES = Path(__file__).resolve().parents[1] / "omnitrain" / "plugins" / "export" / "templates" / "python"
CLASSES = ["dog", "siren", "rain"]
SPEC = AudioSpec(sample_rate=8000, seconds=2.0, n_mels=40, mean=-3.0, std=2.5, clips=3)


def client_module():
    source = (TEMPLATES / "omnitrain_client.py").read_text(encoding="utf-8")
    module = types.ModuleType("omnitrain_client_audio_under_test")
    module.__file__ = str(TEMPLATES / "omnitrain_client.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - our own shipped template
    return module


def audio_meta(spec: AudioSpec = SPEC) -> dict:
    return {
        "task": "classify", "model": "audio_cnn", **spec.to_meta(), "audioColumn": "audio_path",
        "outputName": "class", "names": CLASSES,
    }  # fmt: skip


def weights(spec: AudioSpec = SPEC) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    return rng.normal(0, 1, (spec.n_mels, len(CLASSES))).astype(np.float32), rng.normal(0, 1, len(CLASSES)).astype(
        np.float32
    )


def write_model(path: Path, spec: AudioSpec = SPEC) -> Path:
    """`spectrograms` [K, 1, bands, frames] -> mean over time -> a linear layer: logits [K, classes]. Sensitive to
    every band of the spectrogram it is given."""
    w, b = weights(spec)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["spectrograms"], ["pooled"], axes=[1, 3], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weights"], ["scores"]),
            helper.make_node("Add", ["scores", "bias"], ["logits"]),
        ],
        "tiny_audio",
        [helper.make_tensor_value_info("spectrograms", TensorProto.FLOAT, ["batch", 1, spec.n_mels, spec.frames])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASSES)])],
        [numpy_helper.from_array(w, "weights"), numpy_helper.from_array(b, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def make_client(tmp_path: Path, spec: AudioSpec = SPEC):
    model = write_model(tmp_path / "model.onnx", spec)
    manifest = build_preprocessing_manifest("run-1", {}, audio_meta(spec))
    path = tmp_path / "preprocessing.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return client_module().OmniTrainClient(model, path)


def tone(rate: int, seconds: float, hz: float = 440.0, seed: int = 0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return 0.4 * np.sin(2 * np.pi * hz * t) + np.random.default_rng(seed).normal(0, 0.05, len(t))


def write_wav(path: Path, samples: np.ndarray, rate: int, dtype="<i2") -> Path:
    if np.dtype(dtype).kind == "i":
        samples = np.round(np.clip(samples, -1, 1) * (np.iinfo(dtype).max - 1)).astype(dtype)
    elif np.dtype(dtype).kind == "u":
        samples = np.round(np.clip(samples, -1, 1) * 127 + 128).astype(dtype)
    else:
        samples = samples.astype(dtype)
    wavfile.write(str(path), rate, samples)
    return path


def reference_probabilities(path: Path, spec: AudioSpec = SPEC) -> list[float]:
    """What the backend's own audio code and the same linear map give: the contract the client must reproduce."""
    w, b = weights(spec)
    wave = audio.load_waveform(str(path), spec)
    probabilities = []
    for start in audio.window_starts(len(wave), spec.length, spec.clips):
        mel = audio.spectrogram(audio.cut(wave, start, spec.length), spec)
        logits = mel.mean(axis=1) @ w + b
        e = np.exp(logits - logits.max())
        probabilities.append(e / e.sum())
    return list(np.mean(probabilities, axis=0))


class TestDecoding:
    @pytest.mark.parametrize("dtype", ["<i2", "<i4", "<f4", "<f8", "u1"])
    def test_a_wav_reads_like_the_backend_reads_it(self, tmp_path, dtype):
        module = client_module()
        path = write_wav(tmp_path / "a.wav", tone(8000, 0.5), 8000, dtype)
        samples, rate = module._read_wav(path.read_bytes())  # noqa: SLF001
        expected, expected_rate = audio.decode(str(path))
        assert rate == expected_rate == 8000
        assert samples.shape == expected.shape
        np.testing.assert_allclose(samples, expected, atol=1e-6)

    def test_stereo_has_one_row_per_channel(self, tmp_path):
        module = client_module()
        both = np.stack([tone(8000, 0.25), tone(8000, 0.25, hz=900)], axis=1)
        path = write_wav(tmp_path / "s.wav", both, 8000)
        samples, _ = module._read_wav(path.read_bytes())  # noqa: SLF001
        np.testing.assert_allclose(samples, audio.decode(str(path))[0], atol=1e-6)
        assert samples.shape[0] == 2

    def test_a_24_bit_wav_is_read(self):
        module = client_module()
        values = np.array([0, 1, -1, 8388607, -8388608, 4194304], dtype=np.int32)
        body = b"".join(struct.pack("<i", int(v))[:3] for v in values)
        header = (
            b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 24000, 3, 24) + b"data" + struct.pack("<I", len(body))
        )
        data = b"RIFF" + struct.pack("<I", 4 + len(header) + len(body)) + b"WAVE" + header + body
        samples, rate = module._read_wav(data)  # noqa: SLF001
        assert rate == 8000
        np.testing.assert_allclose(samples[0], values / 8388607.0, atol=1e-7)

    def test_extra_chunks_before_the_data_are_skipped(self, tmp_path):
        module = client_module()
        path = write_wav(tmp_path / "a.wav", tone(8000, 0.1), 8000)
        raw = path.read_bytes()
        # An odd-sized LIST chunk (padded to even) between "fmt " and "data".
        at = raw.index(b"data")
        junk = b"LIST" + struct.pack("<I", 3) + b"abc\x00"
        patched = raw[:at] + junk + raw[at:]
        samples, _ = module._read_wav(patched)  # noqa: SLF001
        np.testing.assert_allclose(samples, audio.decode(str(path))[0], atol=1e-6)

    def test_something_that_is_not_a_wav_raises(self, tmp_path):
        module = client_module()
        with pytest.raises(ValueError):
            module._read_wav(b"not a sound at all")  # noqa: SLF001
        bad = tmp_path / "x.mp3"
        bad.write_bytes(b"not a sound at all")
        with pytest.raises(ValueError):
            module._decode_audio(bad)  # noqa: SLF001


class TestResampling:
    @pytest.mark.parametrize(
        "rate,target",
        [(8000, 16000), (16000, 8000), (22050, 16000), (44100, 16000), (48000, 16000), (11025, 16000), (16000, 16000)],
    )
    def test_it_is_scipys_polyphase_resampler(self, rate, target):
        module = client_module()
        x = tone(rate, 0.6, seed=rate)
        got = module._resample(x, rate, target)  # noqa: SLF001
        expected = resample_poly(x, target // np.gcd(rate, target), rate // np.gcd(rate, target))
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_a_very_short_recording_resamples_too(self):
        module = client_module()
        x = tone(44100, 0.001)
        np.testing.assert_allclose(module._resample(x, 44100, 16000), resample_poly(x, 160, 441), atol=1e-9)  # noqa: SLF001


class TestWindows:
    @pytest.mark.parametrize("total", [0, 1, 500, 1000, 1001, 1500, 3000, 4999, 5000, 12345])
    @pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
    def test_they_start_where_the_backend_takes_them(self, total, count):
        module = client_module()
        assert module._window_starts(total, 1000, count) == audio.window_starts(total, 1000, count)  # noqa: SLF001


class TestSpectrogram:
    def test_the_filterbank_is_the_backends(self):
        module = client_module()
        for n_mels in (40, 64):
            np.testing.assert_allclose(
                module._mel_filterbank(8000, 1024, n_mels, 0.0, 4000.0),  # noqa: SLF001
                audio.mel_filterbank(8000, 1024, n_mels, 0.0, 4000.0),
                atol=1e-12,
            )

    @pytest.mark.parametrize("rate,seconds", [(8000, 5.5), (22050, 3.1), (44100, 2.0), (16000, 0.7)])
    def test_the_batch_is_the_backends_windows_and_spectrograms(self, tmp_path, rate, seconds):
        client = make_client(tmp_path)
        path = write_wav(tmp_path / "a.wav", tone(rate, seconds, seed=rate), rate)
        got = client._audio_spectrograms(path)  # noqa: SLF001
        wave = audio.load_waveform(str(path), SPEC)
        starts = audio.window_starts(len(wave), SPEC.length, SPEC.clips)
        expected = np.stack([audio.spectrogram(audio.cut(wave, s, SPEC.length), SPEC) for s in starts])[:, None]
        assert got.shape == expected.shape == (len(starts), 1, SPEC.n_mels, SPEC.frames) and got.dtype == np.float32
        np.testing.assert_allclose(got, expected, atol=2e-4)

    def test_stereo_is_averaged(self, tmp_path):
        client = make_client(tmp_path)
        one = tone(8000, 2.0)
        path = write_wav(tmp_path / "s.wav", np.stack([one, np.zeros_like(one)], axis=1), 8000)
        got = client._audio_spectrograms(path)  # noqa: SLF001
        wave = audio.load_waveform(str(path), SPEC)
        expected = audio.spectrogram(audio.cut(wave, 0, SPEC.length), SPEC)
        np.testing.assert_allclose(got[0, 0], expected, atol=2e-4)


class TestPredict:
    @pytest.mark.parametrize("rate,seconds", [(8000, 5.5), (22050, 3.1), (44100, 2.0), (16000, 0.7), (8000, 2.0)])
    def test_the_probabilities_match_the_backend(self, tmp_path, rate, seconds):
        client = make_client(tmp_path)
        path = write_wav(tmp_path / "a.wav", tone(rate, seconds, seed=rate), rate)
        got = client.predict(path, top_k=3)
        expected = dict(zip(CLASSES, reference_probabilities(path), strict=True))
        assert set(got) == set(CLASSES)
        for label, probability in got.items():
            assert probability == pytest.approx(expected[label], abs=2e-3)

    def test_the_answer_is_sorted_and_top_k_limits_it(self, tmp_path):
        client = make_client(tmp_path)
        got = client.predict(write_wav(tmp_path / "a.wav", tone(8000, 3.0), 8000), top_k=2)
        assert len(got) == 2 and list(got.values()) == sorted(got.values(), reverse=True)

    def test_a_silent_recording_is_finite(self, tmp_path):
        client = make_client(tmp_path)
        got = client.predict(write_wav(tmp_path / "z.wav", np.zeros(8000), 8000), top_k=3)
        assert all(np.isfinite(p) for p in got.values())

    def test_a_recording_with_no_samples_raises(self, tmp_path):
        client = make_client(tmp_path)
        empty = write_wav(tmp_path / "e.wav", np.zeros(0), 8000)
        with pytest.raises(ValueError):
            client.predict(empty)

    def test_an_unknown_recipe_is_refused(self, tmp_path):
        model = write_model(tmp_path / "model.onnx")
        manifest = build_preprocessing_manifest("run-1", {}, audio_meta())
        manifest["inputs"][0]["audioSpectrogram"]["recipe"] = "log_mel_v9"
        path = tmp_path / "preprocessing.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(NotImplementedError, match="log_mel_v9"):
            client_module().OmniTrainClient(model, path)

    def test_an_audio_input_without_a_recipe_still_says_what_to_do(self, tmp_path):
        model = write_model(tmp_path / "model.onnx")
        manifest = build_preprocessing_manifest("run-1", {}, audio_meta())
        del manifest["inputs"][0]["audioSpectrogram"]
        path = tmp_path / "preprocessing.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        client = client_module().OmniTrainClient(model, path)
        with pytest.raises(NotImplementedError, match="preprocessing.json"):
            client.predict(write_wav(tmp_path / "a.wav", tone(8000, 1.0), 8000))


class TestBundles:
    FORMATS = {
        "python_devkit": "omnitrain_client.py",
        "typescript_devkit": "client.ts",
        "csharp_devkit": "OmniTrainClient.cs",
        "java_devkit": "OmniTrainClient.java",
        "pwa_app": "app.js",
        "flutter_app": "lib/omnitrain_client.dart",
    }

    def test_an_audio_task_needs_the_audio_client_and_every_client_offers_it(self):
        from omnitrain.domain.task_registry import get_task_descriptor
        from omnitrain.plugins.export.formats._kits import client_kind
        from omnitrain.plugins.export.registry import list_export_formats

        task = get_task_descriptor("audio_classification")
        assert client_kind(task) == "audio"
        assert set(self.FORMATS) <= {f.id for f in list_export_formats(task, None)}

    @staticmethod
    def bundle(fmt: str, sample: tuple[str, bytes] = ("input.wav", b"x")) -> dict[str, str]:
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files

        manifest = json.loads(json.dumps(build_preprocessing_manifest("r", {}, audio_meta())))
        files = assemble_files(
            run_name="r", task_label="Audio Classification", format_id=fmt, model_bytes=b"m", preprocessing=manifest,
            golden=GoldenSample(b"{}", *sample),
        )  # fmt: skip
        return {f.path: f.data.decode(errors="replace") for f in files}

    @pytest.mark.parametrize(("fmt", "client_file"), list(FORMATS.items()))
    def test_every_clients_bundle_ships_the_audio_code_and_says_so(self, fmt, client_file):
        files = self.bundle(fmt)
        client = files[client_file]
        for needed in ("audioSpectrogram", "log_mel_v1", "logFloor", "windowsPerAudio"):
            assert needed.lower() in client.lower(), needed
        readme = files["README.md"]
        assert "Audio classification" in readme and "WAV" in readme
        assert "image classification** inputs today" not in readme

    def test_the_python_bundle_verifies_a_real_sample_end_to_end(self, tmp_path):
        import subprocess
        import sys

        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files, to_bundle_local

        wav = write_wav(tmp_path / "input.wav", tone(22050, 3.1, seed=2), 22050)
        probabilities = {k: float(v) for k, v in zip(CLASSES, reference_probabilities(wav), strict=True)}
        expected = {"outputColumn": "class", "outputType": "category", "predictions": probabilities}
        golden = GoldenSample(to_bundle_local(expected, "input.wav"), "input.wav", wav.read_bytes())
        write_model(tmp_path / "model.onnx")
        manifest = json.loads(json.dumps(build_preprocessing_manifest("r", {}, audio_meta())))
        files = assemble_files(
            run_name="r", task_label="t", format_id="python_devkit", model_bytes=(tmp_path / "model.onnx").read_bytes(),
            preprocessing=manifest, golden=golden,
        )  # fmt: skip
        out = tmp_path / "bundle"
        for f in files:
            target = out / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.data)
        ran = subprocess.run(  # noqa: S603 - our own generated script, in a temp folder
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=120
        )
        assert ran.returncode == 0 and ran.stdout.startswith("OK"), ran.stdout + ran.stderr

        # The same bundle with a changed prediction fails.
        (out / "expected.json").write_text(
            json.dumps(
                {
                    **json.loads((out / "expected.json").read_text()),
                    "predictions": {"dog": 0.01, "siren": 0.01, "rain": 0.98},
                }
            )
        )
        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=120
        )
        assert ran.returncode == 1 and "MISMATCH" in ran.stdout

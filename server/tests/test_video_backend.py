"""The video trainer backend without training anything: compile, clip sampling, the manifest, registration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef
from omnitrain.plugins.backends.registry import get_backend, trainable_backends
from omnitrain.plugins.backends.video import VideoBackend, frames
from omnitrain.plugins.backends.video.compile import VideoHyperparameters, compile_video_config, hyperparameter_specs
from omnitrain.plugins.backends.video.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.video.schedule import lr_factor
from omnitrain.plugins.export.formats._kits import supports_generated_clients
from omnitrain.plugins.export.registry import list_export_formats

TASK = get_task_descriptor("video_classification")
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="video_path", kind="storage_uri"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=["run", "walk", "jump"],
)


def compile_(ctx: SnapshotContext = SNAPSHOT, **selection):
    return compile_video_config(TASK, ctx, VideoHyperparameters(**selection))


class TestRegistration:
    def test_the_backend_trains_only_video_classification(self):
        assert get_backend("video") is VideoBackend
        assert VideoBackend.supports(TASK)
        assert not VideoBackend.supports(get_task_descriptor("image_classification"))
        assert VideoBackend in trainable_backends(TASK)

    def test_it_offers_pretrained_models_for_the_task_only(self):
        models = VideoBackend.models(TASK)
        assert [m.id for m in models][:2] == ["mc3_18", "r3d_18"] and all(m.pretrained for m in models)
        assert VideoBackend.models(get_task_descriptor("image_classification")) == []

    def test_listing_it_never_imports_torch(self):
        import subprocess
        import sys

        code = (
            "import sys; from omnitrain.plugins.backends.registry import list_backends; "
            "[b.available() for b in list_backends()]; print('torch' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"

    def test_hyperparameter_specs_cover_every_knob_the_model_accepts(self):
        names = {p.name for p in hyperparameter_specs(TASK)}
        assert {"numFrames", "frameStride", "frameSize", "clipsPerVideo", "pretrained", "freezeBackbone"} <= names
        assert hyperparameter_specs(get_task_descriptor("image_classification")) == []


class TestCompile:
    def test_defaults_come_from_the_first_model(self):
        c = compile_()
        assert (c["backend"], c["task"], c["model"], c["pretrained"]) == ("video", "classify", "mc3_18", True)
        assert (c["clipColumn"], c["labelColumn"], c["classNames"]) == ("video_path", "class", ["run", "walk", "jump"])
        assert c["clip"]["numFrames"] == 16 and c["clip"]["frameSize"] == 112 and c["clip"]["frameStride"] == 2

    def test_a_models_own_input_size_and_normalisation_are_used(self):
        c = compile_(model_id="s3d")
        assert c["clip"]["frameSize"] == 224 and c["clip"]["mean"][0] == pytest.approx(0.43216)
        swin = compile_(model_id="swin3d_t")["clip"]
        assert swin["numFrames"] == 32 and swin["mean"][0] == pytest.approx(0.485)

    def test_seed_is_unset_unless_given(self):
        assert compile_()["train"]["seed"] is None
        assert {s.name: s for s in hyperparameter_specs(TASK)}["seed"].default is None

    def test_explicit_selections_override_defaults(self):
        c = compile_(num_frames=8, frame_size=64, epochs=3, batch_size=2, optimizer="SGD", pretrained=False, seed=7)
        assert (c["clip"]["numFrames"], c["clip"]["frameSize"]) == (8, 64)
        assert (c["train"]["epochs"], c["train"]["batchSize"], c["train"]["optimizer"]) == (3, 2, "SGD")
        assert c["pretrained"] is False and c["train"]["seed"] == 7

    def test_auto_batch_size_falls_back_to_a_small_default(self):
        assert compile_(batch_size="auto")["train"]["batchSize"] == 4

    def test_a_disabled_early_stop_can_never_trigger(self):
        train = compile_(epochs=12, early_stop_patience=-1)["train"]
        assert train["patience"] == 12

    def test_unknown_model_is_a_config_error(self):
        with pytest.raises(ConfigError, match="not a video model"):
            compile_(model_id="resnet50")

    def test_needs_at_least_two_classes(self):
        one = SnapshotContext(columns=SNAPSHOT.columns, label_class_names=["run"])
        with pytest.raises(ConfigError, match="two classes"):
            compile_(one)

    def test_snapshot_without_a_label_column_is_a_config_error(self):
        bare = SnapshotContext(columns=[SNAPSHOT.columns[0]], label_class_names=["a", "b"])
        with pytest.raises(ConfigError, match='"label" column'):
            compile_(bare)

    def test_a_custom_model_of_another_kind_is_refused(self):
        other = CustomModelRef(id="u", kind="timm_image", source_kind="hub", local_path="unused")
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_video_config(TASK, SNAPSHOT, VideoHyperparameters(), custom=other)

    def test_unknown_hyperparameters_are_rejected(self):
        with pytest.raises(ValidationError):
            VideoHyperparameters(imgsz=224)  # type: ignore[call-arg]


SPEC = frames.ClipSpec(num_frames=4, stride=2, size=16, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))


class TestOptimisation:
    def test_defaults_keep_what_each_optimizer_already_did(self):
        train = compile_()["train"]
        assert train["weightDecay"] is None and train["gradClipNorm"] is None
        assert (train["lrSchedule"], train["warmupEpochs"], train["mixedPrecision"]) == ("constant", 0, False)

    def test_selections_reach_the_config(self):
        train = compile_(
            weight_decay=0.05, lr_schedule="cosine", warmup_epochs=1.5, grad_clip_norm=1.0, mixed_precision=True
        )["train"]
        assert (train["weightDecay"], train["lrSchedule"], train["warmupEpochs"]) == (0.05, "cosine", 1.5)
        assert (train["gradClipNorm"], train["mixedPrecision"]) == (1.0, True)

    def test_the_form_offers_every_shape(self):
        specs = {p.name: p for p in hyperparameter_specs(TASK)}
        assert specs["lrSchedule"].choices == ["constant", "linear", "cosine"]
        assert {"weightDecay", "warmupEpochs", "gradClipNorm", "mixedPrecision"} <= set(specs)


class TestLearningRateCurve:
    def test_constant_is_flat(self):
        assert [lr_factor(i, 10, 0, "constant") for i in range(10)] == [1.0] * 10

    def test_linear_starts_at_one_and_ends_at_zero(self):
        curve = [lr_factor(i, 11, 0, "linear") for i in range(11)]
        assert curve[0] == 1.0 and curve[-1] == 0.0
        assert curve == sorted(curve, reverse=True)

    def test_cosine_is_half_way_down_in_the_middle(self):
        assert lr_factor(5, 11, 0, "cosine") == pytest.approx(0.5)
        assert lr_factor(10, 11, 0, "cosine") == pytest.approx(0.0, abs=1e-9)

    def test_warmup_rises_to_the_full_rate_then_the_schedule_takes_over(self):
        curve = [lr_factor(i, 20, 4, "linear") for i in range(20)]
        assert curve[:4] == sorted(curve[:4]) and curve[0] < 0.25
        assert curve[4] == 1.0 and curve[-1] == 0.0

    def test_a_step_past_the_end_stays_at_the_floor(self):
        assert lr_factor(99, 10, 0, "linear") == 0.0

    def test_an_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            lr_factor(0, 10, 0, "zigzag")


class TestStepper:
    """The optimizer step with its schedule, clipping and (CUDA-only) mixed precision, on a toy network on the CPU."""

    @staticmethod
    def _stepper(**hp):
        import torch

        from omnitrain.plugins.backends.video.train import _Stepper

        net = torch.nn.Linear(4, 2)
        params = list(net.parameters())
        optimizer = torch.optim.SGD(params, lr=1.0)
        base = {"epochs": 2, "lrSchedule": "linear", "warmupEpochs": 0, "gradClipNorm": None, "mixedPrecision": True}
        return net, optimizer, _Stepper(optimizer, params, "cpu", {**base, **hp}, steps_per_epoch=5)

    def test_mixed_precision_is_off_without_cuda(self):
        _, _, stepper = self._stepper()
        assert stepper.amp is False

    def test_the_learning_rate_follows_the_curve_per_step(self):
        import torch

        net, optimizer, stepper = self._stepper()
        rates = []
        for _ in range(10):
            rates.append(optimizer.param_groups[0]["lr"])
            stepper.step(net(torch.ones(1, 4)).sum())
        assert rates[0] == 1.0 and rates == sorted(rates, reverse=True)

    def test_clipping_caps_the_update(self):
        import torch

        net, _, stepper = self._stepper(gradClipNorm=0.1, lrSchedule="constant")
        before = [p.detach().clone() for p in net.parameters()]
        stepper.step(net(torch.ones(1, 4) * 100).sum() * 100)
        moved = torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(net.parameters(), before, strict=True)))
        assert float(moved.detach()) <= 0.1 + 1e-5


class TestClipSampling:
    def test_a_clip_spans_frames_times_stride(self):
        assert SPEC.span == 7

    def test_scoring_clips_are_spread_evenly_over_the_video(self):
        assert frames.clip_starts(frame_count=27, spec=SPEC, clips=3) == [0, 10, 20]
        assert frames.clip_starts(frame_count=27, spec=SPEC, clips=1) == [10]

    def test_a_random_start_stays_inside_the_video(self):
        rng = np.random.default_rng(0)
        starts = {frames.clip_starts(27, SPEC, 1, rng)[0] for _ in range(200)}
        assert min(starts) >= 0 and max(starts) <= 20 and len(starts) > 5

    def test_a_short_video_repeats_its_last_frame(self):
        assert frames.clip_starts(frame_count=3, spec=SPEC, clips=3) == [0, 0, 0]
        assert frames.clip_indices(0, frame_count=3, spec=SPEC) == [0, 2, 2, 2]

    def test_indices_are_stride_apart(self):
        assert frames.clip_indices(5, frame_count=100, spec=SPEC) == [5, 7, 9, 11]


def _clip_file(directory: str, frames_n: int = 20, width: int = 48, height: int = 32) -> str:
    import cv2

    path = str(Path(directory) / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for i in range(frames_n):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = min(255, 10 * i)  # brightness grows with the frame number, so order is checkable
        writer.write(frame)
    writer.release()
    return path


class TestLoadClips:
    def test_clips_are_square_and_have_the_requested_frame_count(self):
        with tempfile.TemporaryDirectory() as d:
            clips = frames.load_clips(_clip_file(d), SPEC, clips=3)
        assert len(clips) == 3 and all(c.shape == (4, 16, 16, 3) and c.dtype == np.uint8 for c in clips)

    def test_frames_come_in_video_order(self):
        with tempfile.TemporaryDirectory() as d:
            [clip] = frames.load_clips(_clip_file(d), SPEC, clips=1)
        green = clip[:, 8, 8, 1].astype(int)  # RGB, so channel 1 is the one that grows
        assert list(green) == sorted(green) and green[-1] > green[0]

    def test_later_clips_start_later_in_the_video(self):
        with tempfile.TemporaryDirectory() as d:
            clips = frames.load_clips(_clip_file(d), SPEC, clips=3)
        means = [int(c[:, :, :, 1].mean()) for c in clips]
        assert means == sorted(means) and means[0] < means[-1]

    def test_flip_mirrors_left_to_right(self):
        with tempfile.TemporaryDirectory() as d:
            path = _clip_file(d)
            [plain] = frames.load_clips(path, SPEC, clips=1)
            [flipped] = frames.load_clips(path, SPEC, clips=1, flip=True)
        assert np.array_equal(flipped, plain[:, :, ::-1])

    def test_a_random_crop_is_one_place_for_the_whole_clip(self):
        with tempfile.TemporaryDirectory() as d:
            path = _clip_file(d, width=80, height=32)
            [clip] = frames.load_clips(path, SPEC, rng=np.random.default_rng(3))
        assert clip.shape == (4, 16, 16, 3)

    def test_a_file_that_is_not_a_video_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.mp4"
            path.write_bytes(b"nope" * 100)
            with pytest.raises(ValueError):
                frames.load_clips(str(path), SPEC)

    def test_to_tensor_is_channels_first_and_normalised(self):
        torch = pytest.importorskip("torch")
        clip = np.full((4, 16, 16, 3), 255, dtype=np.uint8)
        spec = frames.ClipSpec(4, 2, 16, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
        tensor = frames.to_tensor(clip, spec)
        assert tensor.shape == (3, 4, 16, 16) and tensor.dtype == torch.float32
        assert float(tensor.mean()) == pytest.approx(2.0)  # (1.0 - 0.5) / 0.25


class TestManifestAndExport:
    META = {
        "task": "classify", "model": "mc3_18", "numFrames": 16, "frameStride": 2, "frameSize": 112,
        "mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6], "clipsPerVideo": 3, "clipColumn": "video_path",
        "outputName": "class", "names": ["walk", "run"],
    }  # fmt: skip

    def test_the_manifest_describes_sampling_resize_normalisation_and_the_models_own_classes(self):
        m = build_preprocessing_manifest("r1", {}, self.META)
        [inp] = m["inputs"]
        assert inp["type"] == "video" and inp["layout"] == "NCTHW"
        assert inp["videoSampling"] == {
            "numFrames": 16, "frameStride": 2, "clipsPerVideo": 3,
            "clipStarts": "evenly_spaced", "shortVideo": "repeat_last_frame",
        }  # fmt: skip
        assert inp["imageResize"]["mode"] == "shorter_edge_center_crop"
        assert inp["imageNormalization"] == {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
        assert m["outputs"][0]["classes"] == ["walk", "run"]

    def test_the_manifest_needs_the_training_metadata(self):
        with pytest.raises(ValueError, match="names.json"):
            build_preprocessing_manifest("r1", {}, None)

    def test_the_bare_model_and_the_python_devkit_are_offered(self):
        assert not supports_generated_clients(TASK)
        ids = [f.id for f in list_export_formats(TASK, VideoBackend)]
        assert ids == ["onnx", "python_devkit"]

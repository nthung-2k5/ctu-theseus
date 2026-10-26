"""The generated Python devkit client for a detector and for a video classifier.

The ONNX models here are built by hand, so what is under test is the client's own arithmetic and nothing else: a fake
detector that returns a known candidate table (to check the letterbox, the confidence filter, non-maximum suppression
and the mapping back to the original image) and a tiny video "classifier" (to check frame sampling and normalisation
against the backend's own `frames` module, which is the reference the manifest describes).
"""

import types
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from omnitrain.plugins.backends.video import frames
from omnitrain.plugins.backends.video.manifest import build_preprocessing_manifest as video_manifest
from omnitrain.plugins.backends.yolo.manifest import build_preprocessing_manifest as yolo_manifest

TEMPLATES = Path(__file__).resolve().parents[1] / "omnitrain" / "plugins" / "export" / "templates" / "python"


def client_class():
    """The devkit client, executed from its template source. Not imported as a module: that would write a bytecode
    cache into the templates folder, which is shipped as is."""
    source = (TEMPLATES / "omnitrain_client.py").read_text(encoding="utf-8")
    module = types.ModuleType("omnitrain_client_under_test")
    module.__file__ = str(TEMPLATES / "omnitrain_client.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - our own shipped template
    return module.OmniTrainClient


def write_manifest(path: Path, manifest: dict) -> Path:
    import json

    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# -- Detection ----------------------------------------------------------------------------------------------

SIZE = 64
CLASSES = ["cat", "dog"]


def detector_model(path: Path, candidates: np.ndarray) -> Path:
    """An ONNX graph shaped like an exported YOLO detector: input `images` [1, 3, S, S], output `output0`
    [1, 4 + classes, candidates], whatever the image. `candidates` is [N, 4 + classes] (cx, cy, w, h, scores...)."""
    table = np.ascontiguousarray(candidates.T[np.newaxis]).astype(np.float32)
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")
    nodes = [
        # The output must depend on the input for the graph to keep it; multiplying a sum by zero does that for free.
        helper.make_node("ReduceSum", ["images"], ["total"], keepdims=0),
        helper.make_node("Mul", ["total", "zero"], ["nothing"]),
        helper.make_node("Add", ["table", "nothing"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fake_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(table.shape))],
        [numpy_helper.from_array(table, "table"), zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def detector_meta() -> dict:
    return {
        "task": "detect", "imgsz": SIZE, "imageColumn": "image", "outputName": "boxes", "names": CLASSES,
    }  # fmt: skip


def detector(tmp_path: Path, candidates: list[list[float]]):
    model = detector_model(tmp_path / "model.onnx", np.array(candidates, dtype=np.float32))
    manifest = write_manifest(tmp_path / "preprocessing.json", yolo_manifest("run-1", {}, detector_meta()))
    return client_class()(model, manifest)


@pytest.fixture
def wide_image(tmp_path) -> Path:
    """200 x 100: scaled by 0.32 to 64 x 32, so padded 16 rows above and below."""
    path = tmp_path / "wide.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 200, 3), dtype=np.uint8)).save(path)
    return path


class TestDetection:
    def test_a_box_is_mapped_back_to_fractions_of_the_original_image(self, tmp_path, wide_image):
        # Image box (50, 20)-(150, 80) is, letterboxed at scale 0.32 with 16 rows of padding: x 16..48, y 22.4..41.6.
        client = detector(tmp_path, [[32.0, 32.0, 32.0, 19.2, 0.9, 0.1]])
        [found] = client.predict(wide_image)
        assert found["label"] == "cat" and found["confidence"] == pytest.approx(0.9)
        assert found["box"] == pytest.approx({"x": 0.25, "y": 0.2, "w": 0.5, "h": 0.6}, abs=1e-5)

    def test_a_less_confident_duplicate_of_the_same_class_is_suppressed(self, tmp_path, wide_image):
        client = detector(
            tmp_path,
            [
                [32.0, 32.0, 32.0, 19.2, 0.80, 0.0],
                [32.5, 32.0, 32.0, 19.2, 0.90, 0.0],  # nearly the same box, more confident: it wins
            ],
        )
        found = client.predict(wide_image)
        assert [d["confidence"] for d in found] == [pytest.approx(0.9)]

    def test_an_overlapping_box_of_another_class_is_kept(self, tmp_path, wide_image):
        client = detector(tmp_path, [[32.0, 32.0, 32.0, 19.2, 0.9, 0.0], [32.0, 32.0, 32.0, 19.2, 0.0, 0.7]])
        assert [d["label"] for d in client.predict(wide_image)] == ["cat", "dog"]

    def test_two_separate_boxes_of_one_class_are_both_kept_most_confident_first(self, tmp_path, wide_image):
        client = detector(tmp_path, [[12.0, 32.0, 8.0, 8.0, 0.4, 0.0], [52.0, 32.0, 8.0, 8.0, 0.8, 0.0]])
        assert [d["confidence"] for d in client.predict(wide_image)] == [pytest.approx(0.8), pytest.approx(0.4)]

    def test_a_box_below_the_confidence_threshold_is_dropped_and_the_threshold_can_be_changed(
        self, tmp_path, wide_image
    ):
        client = detector(tmp_path, [[32.0, 32.0, 10.0, 10.0, 0.2, 0.1]])
        assert client.predict(wide_image) == []
        assert len(client.detect(wide_image, confidence=0.15)) == 1

    def test_the_overlap_threshold_can_be_changed(self, tmp_path, wide_image):
        rows = [[28.0, 32.0, 16.0, 16.0, 0.9, 0.0], [34.0, 32.0, 16.0, 16.0, 0.8, 0.0]]  # overlap ~0.45
        client = detector(tmp_path, rows)
        assert len(client.detect(wide_image, iou=0.7)) == 2
        assert len(client.detect(wide_image, iou=0.3)) == 1

    def test_a_box_that_runs_off_the_image_is_clipped_to_it(self, tmp_path, wide_image):
        # Reaches into the padding above the image (letterboxed y 6..26 is rows 16 and up of the picture).
        client = detector(tmp_path, [[32.0, 16.0, 20.0, 20.0, 0.9, 0.0]])
        [found] = client.predict(wide_image)
        assert found["box"]["y"] == 0.0
        assert 0 <= found["box"]["x"] and found["box"]["x"] + found["box"]["w"] <= 1.0
        assert found["box"]["y"] + found["box"]["h"] <= 1.0

    def test_max_detections_caps_the_list(self, tmp_path, wide_image):
        rows = [[6.0 + 10 * i, 32.0, 6.0, 6.0, 0.9 - 0.01 * i, 0.0] for i in range(5)]
        assert len(detector(tmp_path, rows).detect(wide_image, max_detections=3)) == 3

    def test_no_candidates_means_no_detections(self, tmp_path, wide_image):
        assert detector(tmp_path, [[32.0, 32.0, 10.0, 10.0, 0.0, 0.0]]).predict(wide_image) == []

    def test_the_letterbox_is_ultralytics_own_arithmetic(self, tmp_path):
        """Same padding and scale as `ultralytics.data.augment.LetterBox`, for several awkward sizes."""
        from omnitrain.plugins.backends.yolo import runtime

        runtime.configure()
        from ultralytics.data.augment import LetterBox

        client = detector(tmp_path, [[32.0, 32.0, 10.0, 10.0, 0.0, 0.0]])
        for width, height in [(200, 100), (100, 200), (101, 37), (64, 64), (33, 150), (640, 480)]:
            path = tmp_path / f"{width}x{height}.png"
            rgb = np.full((height, width, 3), 200, dtype=np.uint8)
            Image.fromarray(rgb).save(path)
            tensor, scale, left, top, w0, h0 = client._letterbox(path)  # noqa: SLF001 - the arithmetic is the point
            reference = LetterBox((SIZE, SIZE), auto=False, center=True)(image=rgb[:, :, ::-1].copy())
            assert tensor.shape == (1, 3, SIZE, SIZE) and (w0, h0) == (width, height)
            # Where the picture ends and the grey padding begins must match to the pixel.
            ours = (np.abs(tensor[0, 0] * 255 - 200) < 1).nonzero()
            theirs = (np.abs(reference[:, :, 0].astype(float) - 200) < 1).nonzero()
            assert (ours[0].min(), ours[0].max(), ours[1].min(), ours[1].max()) == (
                theirs[0].min(), theirs[0].max(), theirs[1].min(), theirs[1].max()
            ), (width, height)  # fmt: skip
            assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255) or (left == 0 and top == 0)

    def test_a_manifest_with_another_resize_mode_is_refused(self, tmp_path, wide_image):
        manifest = yolo_manifest("run-1", {}, detector_meta())
        manifest["inputs"][0]["imageResize"] = {"mode": "something_new"}
        model = detector_model(tmp_path / "model.onnx", np.zeros((1, 6), dtype=np.float32))
        client = client_class()(model, write_manifest(tmp_path / "preprocessing.json", manifest))
        with pytest.raises(NotImplementedError, match="something_new"):
            client.predict(wide_image)


# -- Video --------------------------------------------------------------------------------------------------

NUM_FRAMES, STRIDE, FRAME_SIZE, CLIPS = 4, 2, 16, 3
MEAN, STD = (0.43, 0.40, 0.37), (0.23, 0.22, 0.22)
VIDEO_CLASSES = ["run", "walk", "jump"]


def video_meta() -> dict:
    return {
        "task": "classify", "frameSize": FRAME_SIZE, "numFrames": NUM_FRAMES, "frameStride": STRIDE,
        "clipsPerVideo": CLIPS, "clipColumn": "video_path", "outputName": "class", "names": VIDEO_CLASSES,
        "mean": list(MEAN), "std": list(STD),
    }  # fmt: skip


def video_weights() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    return rng.normal(0, 3, (3, len(VIDEO_CLASSES))).astype(np.float32), rng.normal(0, 1, len(VIDEO_CLASSES)).astype(
        np.float32
    )


def video_model(path: Path) -> Path:
    """`clips` [K, 3, T, S, S] -> mean over time and space -> a linear layer: logits [K, classes]. Sensitive to the
    colour of every frame it is given, so a wrong frame, crop or normalisation changes its answer."""
    weights, bias = video_weights()
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["clips"], ["pooled"], axes=[2, 3, 4], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weights"], ["scores"]),
            helper.make_node("Add", ["scores", "bias"], ["logits"]),
        ],
        "tiny_video",
        [helper.make_tensor_value_info("clips", TensorProto.FLOAT, ["batch", 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(VIDEO_CLASSES)])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def write_video(path: Path, frame_count: int, width: int = 48, height: int = 32) -> Path:
    """Every frame a different flat colour, so which frames were picked shows in the output."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = ((i * 37) % 255, (i * 91) % 255, (i * 53) % 255)
        # A gradient, so the crop position matters as well.
        frame[:, : width // 2] //= 2
        writer.write(frame)
    writer.release()
    return path


def reference_probabilities(path: Path) -> list[float]:
    """What the backend's own frame code and the same linear map give: the contract the client must reproduce."""
    spec = frames.ClipSpec(NUM_FRAMES, STRIDE, FRAME_SIZE, MEAN, STD)
    weights, bias = video_weights()
    probabilities = []
    for clip in frames.load_clips(str(path), spec, CLIPS):
        pooled = frames.to_tensor(clip, spec).numpy().mean(axis=(1, 2, 3))
        logits = pooled @ weights + bias
        e = np.exp(logits - logits.max())
        probabilities.append(e / e.sum())
    return list(np.mean(probabilities, axis=0))


@pytest.fixture
def video_client(tmp_path):
    model = video_model(tmp_path / "model.onnx")
    manifest = write_manifest(tmp_path / "preprocessing.json", video_manifest("run-1", {}, video_meta()))
    return client_class()(model, manifest)


class TestVideo:
    @pytest.mark.parametrize("frame_count", [40, 9, 3, 1])
    def test_the_probabilities_match_the_backends_own_frame_pipeline(self, tmp_path, video_client, frame_count):
        path = write_video(tmp_path / "clip.avi", frame_count)
        got = video_client.predict(path, top_k=3)
        expected = dict(zip(VIDEO_CLASSES, reference_probabilities(path), strict=True))
        assert set(got) == set(VIDEO_CLASSES)
        for label, probability in got.items():
            assert probability == pytest.approx(expected[label], abs=2e-3)

    def test_a_portrait_video_is_cropped_like_the_backend_crops_it(self, tmp_path, video_client):
        path = write_video(tmp_path / "tall.avi", 30, width=32, height=64)
        got = video_client.predict(path, top_k=3)
        expected = dict(zip(VIDEO_CLASSES, reference_probabilities(path), strict=True))
        assert all(got[k] == pytest.approx(expected[k], abs=2e-3) for k in VIDEO_CLASSES)

    @pytest.mark.parametrize("width,height", [(50, 34), (37, 61), (71, 45), (17, 100)])
    def test_sizes_that_do_not_scale_to_whole_pixels_round_the_way_the_backend_does(
        self, tmp_path, video_client, width, height
    ):
        path = write_video(tmp_path / "odd.avi", 24, width=width, height=height)
        got = video_client.predict(path, top_k=3)
        expected = dict(zip(VIDEO_CLASSES, reference_probabilities(path), strict=True))
        assert all(got[k] == pytest.approx(expected[k], abs=2e-3) for k in VIDEO_CLASSES)

    def test_the_answer_is_sorted_and_top_k_limits_it(self, tmp_path, video_client):
        got = video_client.predict(write_video(tmp_path / "clip.avi", 30), top_k=2)
        assert len(got) == 2 and list(got.values()) == sorted(got.values(), reverse=True)

    def test_a_file_that_is_not_a_video_raises(self, tmp_path, video_client):
        bogus = tmp_path / "bogus.avi"
        bogus.write_bytes(b"not a video")
        with pytest.raises(ValueError):
            video_client.predict(bogus)

    def test_the_clips_are_batched_in_the_manifests_layout(self, tmp_path, video_client):
        clips = video_client._video_clips(write_video(tmp_path / "clip.avi", 40))  # noqa: SLF001
        assert clips.shape == (CLIPS, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE) and clips.dtype == np.float32


# -- What the export system offers and ships ---------------------------------------------------------------


class TestBundles:
    @pytest.mark.parametrize(
        ("task", "kind"),
        [
            ("image_classification", "standard"),
            ("tabular_regression", "standard"),
            ("object_detection", "detection"),
            ("video_classification", "video"),
        ],
    )
    def test_a_task_needs_the_client_kind_its_output_and_modality_say(self, task, kind):
        from omnitrain.domain.task_registry import get_task_descriptor
        from omnitrain.plugins.export.formats._kits import client_kind

        assert client_kind(get_task_descriptor(task)) == kind

    def test_every_client_offers_detection_and_only_the_python_devkit_offers_video(self):
        from omnitrain.domain.task_registry import get_task_descriptor
        from omnitrain.plugins.export.registry import list_export_formats

        clients = {"python_devkit", "typescript_devkit", "csharp_devkit", "java_devkit", "pwa_app", "flutter_app"}
        detection = {f.id for f in list_export_formats(get_task_descriptor("object_detection"), None)}
        assert clients <= detection
        video = {f.id for f in list_export_formats(get_task_descriptor("video_classification"), None)}
        assert video & clients == {"python_devkit"}
        standard = {f.id for f in list_export_formats(get_task_descriptor("image_classification"), None)}
        assert {"python_devkit", "typescript_devkit", "csharp_devkit", "java_devkit", "pwa_app"} <= standard

    def test_the_bundles_expected_json_keeps_a_detectors_boxes(self):
        import json

        from omnitrain.plugins.export.bundle import to_bundle_local

        boxes = [{"label": "cat", "confidence": 0.9, "box": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]
        expected = {"outputColumn": "boxes", "outputType": "detections", "predictions": {"cat": 0.9}}
        kept = json.loads(to_bundle_local({**expected, "detections": boxes}, "input.png"))
        assert kept["detections"] == boxes and kept["sampleFile"] == "sample/input.png"
        assert "detections" not in json.loads(to_bundle_local(expected, "input.png"))

    def test_the_readme_says_what_a_detector_and_a_video_model_need(self):
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files

        def readme(manifest: dict) -> str:
            files = assemble_files(
                run_name="r", task_label="t", format_id="python_devkit", model_bytes=b"m",
                preprocessing=manifest, golden=GoldenSample(b"{}", "input.bin", b"x"),
            )  # fmt: skip
            return next(f.data for f in files if f.path == "README.md").decode()

        detection = readme(yolo_manifest("r", {}, detector_meta()))
        assert "non-maximum suppression" in detection and "opencv" not in detection
        video = readme(video_manifest("r", {}, video_meta()))
        assert "pip install onnxruntime numpy pillow opencv-python-headless" in video
        assert "non-maximum suppression" not in video

    @pytest.mark.parametrize(
        ("fmt", "client_file"),
        [
            ("python_devkit", "omnitrain_client.py"),
            ("typescript_devkit", "client.ts"),
            ("csharp_devkit", "OmniTrainClient.cs"),
            ("java_devkit", "OmniTrainClient.java"),
            ("pwa_app", "app.js"),
            ("flutter_app", "lib/omnitrain_client.dart"),
        ],
    )
    def test_every_clients_bundle_for_a_detector_ships_the_detection_code_and_says_so(self, fmt, client_file):
        import re

        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files

        files = {
            f.path: f.data.decode(errors="replace")
            for f in assemble_files(
                run_name="r", task_label="t", format_id=fmt, model_bytes=b"m",
                preprocessing=yolo_manifest("r", {}, detector_meta()), golden=GoldenSample(b"{}", "input.bin", b"x"),
            )
        }  # fmt: skip
        client = files[client_file]
        assert re.search(r"letterbox", client, re.I) and re.search(r"non_?max_?suppression", client, re.I)
        assert "non-maximum suppression" in files["README.md"]
        assert "image classification** inputs today" not in files["README.md"]

    @staticmethod
    def detector_bundle(
        tmp_path: Path, expected_boxes: list[dict], candidates: list[list[float]], fmt: str = "python_devkit"
    ) -> Path:
        """A real devkit bundle for a fake detector, written out and ready to run its verify script in."""
        import json

        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files, to_bundle_local

        model_path = detector_model(tmp_path / "fake.onnx", np.array(candidates, dtype=np.float32))
        expected = {
            "outputColumn": "boxes", "outputType": "detections", "predictions": {}, "detections": expected_boxes,
        }  # fmt: skip
        image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 200, 3), dtype=np.uint8))
        png = tmp_path / "input.png"
        image.save(png)
        golden = GoldenSample(to_bundle_local(expected, "input.png"), "input.png", png.read_bytes())
        files = assemble_files(
            run_name="r", task_label="t", format_id=fmt, model_bytes=model_path.read_bytes(),
            preprocessing=json.loads(json.dumps(yolo_manifest("r", {}, detector_meta()))), golden=golden,
        )  # fmt: skip
        out = tmp_path / "bundle"
        for f in files:
            target = out / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.data)
        return out

    @staticmethod
    def run(bundle: Path, *command: str):
        import subprocess
        import sys

        return subprocess.run(  # noqa: S603 - our own generated script, in a temp folder
            [sys.executable, "-B", *command], cwd=bundle, capture_output=True, text=True, timeout=120
        )

    def test_verify_passes_when_the_clients_boxes_match_the_platforms(self, tmp_path):
        # The 200 x 100 sample at scale 0.32: this candidate is the image box x 0.25..0.75, y 0.2..0.8.
        boxes = [{"label": "cat", "confidence": 0.9, "box": {"x": 0.25, "y": 0.2, "w": 0.5, "h": 0.6}}]
        bundle = self.detector_bundle(tmp_path, boxes, [[32.0, 32.0, 32.0, 19.2, 0.88, 0.1]])
        done = self.run(bundle, "verify.py")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "OK" in done.stdout

    def test_verify_fails_when_the_platform_saw_something_the_client_missed(self, tmp_path):
        boxes = [{"label": "dog", "confidence": 0.9, "box": {"x": 0.25, "y": 0.2, "w": 0.5, "h": 0.6}}]
        bundle = self.detector_bundle(tmp_path, boxes, [[32.0, 32.0, 32.0, 19.2, 0.88, 0.1]])
        done = self.run(bundle, "verify.py")
        assert done.returncode == 1
        assert "MISMATCH" in done.stdout

    def test_verify_ignores_a_borderline_box_only_one_side_found(self, tmp_path):
        bundle = self.detector_bundle(tmp_path, [], [[32.0, 32.0, 32.0, 19.2, 0.28, 0.1]])
        assert self.run(bundle, "verify.py").returncode == 0

    def test_example_prints_a_line_per_box(self, tmp_path):
        boxes = [{"label": "cat", "confidence": 0.9, "box": {"x": 0.25, "y": 0.2, "w": 0.5, "h": 0.6}}]
        bundle = self.detector_bundle(tmp_path, boxes, [[32.0, 32.0, 32.0, 19.2, 0.88, 0.1]])
        done = self.run(bundle, "example.py", str(bundle / "sample" / "input.png"))
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith("cat: 0.8800  x=0.250 y=0.200 w=0.500 h=0.600")

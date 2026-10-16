"""The video modality's data side: the registry entry, probing a clip, the upload route, poster frames, health."""

import tempfile
from pathlib import Path

import pytest

from omnitrain.api.routers.datasets import upload as datasets_router
from omnitrain.domain import task_registry as tr
from omnitrain.domain import video as video_media
from omnitrain.services import storage


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    store: dict = {}
    store_puts: list = []

    def put(b, k, data, content_type=None):
        store_puts.append((b, k))
        store[(b, k)] = data

    monkeypatch.setattr(storage, "upload_bytes", put)
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    return store_puts


async def project(c, task):
    return (await c.post("/api/projects", json={"name": "p", "description": None, "task": task})).json()["project"]


async def listing(c, pid):
    r = await c.get(f"/api/projects/{pid}/items")
    assert r.status_code == 200, r.text
    return r.json()


def make_mp4(frames: int = 12, width: int = 64, height: int = 48, fps: int = 6) -> bytes:
    """A real, decodable MPEG-4 clip: a block sweeping across the frame."""
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        assert writer.isOpened(), "this OpenCV build cannot write mp4"
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = (i * 4) % (width - 8)
            frame[10:30, x : x + 8] = (0, 255, 0)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


def test_video_classification_is_registered_like_the_other_file_classifiers():
    d = tr.get_task_descriptor("video_classification")
    assert d.modality == "video" and d.status == "experimental"
    assert d.item_spec.payload == "file" and d.item_spec.accept == ["video/mp4", "video/webm"]
    assert [c.name for c in d.columns] == ["video_path", "class", "split"]
    assert tr.class_annotation_type("video_classification") == "classification"
    assert tr.get_inference_input_spec("video_classification")["kind"] == "file"
    assert tr.registry_json()["tasks"]["video_classification"]["inferenceOutputKind"] == "classification"


def test_probe_reads_a_clips_properties():
    info = video_media.probe(make_mp4(frames=12, width=64, height=48, fps=6), ".mp4", "mp4")
    assert (info.width, info.height, info.frame_count, info.fps) == (64, 48, 12, 6.0)
    assert info.duration_seconds == pytest.approx(2.0)
    assert info.container == "mp4"


def test_probe_refuses_a_file_that_is_not_a_video():
    with pytest.raises(video_media.VideoError):
        video_media.probe(b"this is not a video" * 50, ".mp4")


def test_poster_frame_is_a_jpeg_fitted_inside_the_requested_size():
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.mp4"
        path.write_bytes(make_mp4(width=128, height=64))
        jpeg = video_media.poster_jpeg(str(path), 32)
    assert jpeg[:2] == b"\xff\xd8"
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (16, 32)  # the 2:1 clip scaled down to fit 32 wide, aspect kept


def test_container_comes_from_the_content_type_then_the_extension():
    assert video_media.container_for("video/webm", ".mp4") == "webm"
    assert video_media.container_for(None, ".webm") == "webm"
    assert video_media.container_for("video/mp4", "") == "mp4"
    assert video_media.container_for("application/octet-stream", ".mkv") is None


async def upload(c, pid, *files, split="train", **form):
    return await c.post(f"/api/projects/{pid}/upload", data={"split": split, **form}, files=list(files))


async def test_uploading_a_clip_records_its_features_and_a_poster_thumbnail(client, new_user):
    c = await new_user()
    p = await project(c, "video_classification")
    r = await upload(c, p["id"], ("files", ("walk.mp4", make_mp4(frames=12, fps=6), "video/mp4")))
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["status"] == "fulfilled", result

    item = (await listing(c, p["id"]))["items"][0]
    f = item["videoFeatures"]
    assert (f["width"], f["height"], f["frameCount"], f["container"]) == (64, 48, 12, "mp4")
    assert float(f["fps"]) == 6.0 and float(f["durationSeconds"]) == pytest.approx(2.0)
    assert item["thumbnailUrl"] and item["visionFeatures"] is None and item["audioFeatures"] is None


async def test_an_undecodable_file_is_rejected_without_failing_the_batch(client, new_user):
    c = await new_user()
    p = await project(c, "video_classification")
    r = await upload(
        c, p["id"],
        ("files", ("ok.mp4", make_mp4(), "video/mp4")),
        ("files", ("bad.mp4", b"not a video at all" * 40, "video/mp4")),
    )  # fmt: skip
    ok, bad = r.json()["results"]
    assert ok["status"] == "fulfilled"
    assert bad["status"] == "rejected" and "video" in bad["reason"].lower()
    assert (await listing(c, p["id"]))["total"] == 1


async def test_the_video_size_limit_is_its_own_and_larger_than_the_image_limit(client, new_user, monkeypatch):
    assert datasets_router.MAX_VIDEO_UPLOAD_BYTES > datasets_router.MAX_UPLOAD_BYTES
    monkeypatch.setattr(datasets_router, "MAX_VIDEO_UPLOAD_BYTES", 100)
    c = await new_user()
    p = await project(c, "video_classification")
    r = await upload(c, p["id"], ("files", ("big.mp4", make_mp4(), "video/mp4")))
    big = r.json()["results"][0]
    assert big["status"] == "rejected" and "200 MB" in big["reason"]


async def test_re_uploading_the_same_clip_dedups_to_one_item(client, new_user, fake_s3):
    c = await new_user()
    p = await project(c, "video_classification")
    clip = make_mp4()
    first = (await upload(c, p["id"], ("files", ("a.mp4", clip, "video/mp4")))).json()["results"][0]["value"]
    second = (await upload(c, p["id"], ("files", ("b.mp4", clip, "video/mp4")))).json()["results"][0]["value"]
    assert second["id"] == first["id"] and second["isDuplicate"] is True
    assert len(fake_s3) == 1


async def test_the_health_report_describes_the_clips(client, new_user):
    c = await new_user()
    p = await project(c, "video_classification")
    await upload(
        c, p["id"],
        ("files", ("a.mp4", make_mp4(frames=6, fps=6), "video/mp4")),
        ("files", ("b.mp4", make_mp4(frames=18, fps=6), "video/mp4")),
    )  # fmt: skip
    health = (await c.get(f"/api/projects/{p['id']}/dataset/health")).json()["health"]
    assert health["modality"] == "video" and health["itemCount"] == 2
    v = health["video"]
    assert v["count"] == 2 and v["containers"] == {"mp4": 2}
    assert (v["durationSeconds"]["min"], v["durationSeconds"]["max"]) == (1.0, 3.0)
    assert health["vision"] is None and health["audio"] is None

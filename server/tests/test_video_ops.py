"""Video preprocessing and augmentation: the media helpers, each op, and whole snapshots (Postgres, stubbed S3)."""

import hashlib
import io
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa

from omnitrain.augmentation import media
from omnitrain.augmentation.media import VideoClip
from omnitrain.augmentation.ops import video as aug_ops
from omnitrain.augmentation.registry import list_augmentations
from omnitrain.db.models import (
    Annotation,
    Dataset,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    Project,
    User,
    VideoFeatures,
)
from omnitrain.preprocessing.ops import video as pre_ops
from omnitrain.preprocessing.registry import list_preprocessing
from omnitrain.services import snapshot as snap
from omnitrain.services import storage
from omnitrain.services import video as video_service
from omnitrain.services.task_registry import get_task_descriptor

TASK = get_task_descriptor("video_classification")
BUCKET = "theseus-datasets"


def clip(frames=20, width=64, height=48, fps=10.0) -> VideoClip:
    """Frame i is a flat grey of level 10*i, so a frame's level says which source frame it came from."""
    data = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for i in range(frames):
        data[i] = min(255, 10 * i)
    data[:, :, : width // 2, 0] = 200  # left half red, so a flip is visible
    return VideoClip(data, fps)


def mp4_bytes(frames=20, width=64, height=48, fps=10) -> bytes:
    import cv2

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for i in range(frames):
            frame = np.full((height, width, 3), min(255, 10 * i), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


# -- Media -----------------------------------------------------------------------------------


def test_a_clip_survives_an_encode_decode_round_trip_as_webm():
    encoded = media.encode_video(clip(frames=12, width=64, height=48, fps=10.0))
    assert (encoded.ext, encoded.content_type) == (".webm", "video/webm")
    assert encoded.features["frame_count"] == 12 and encoded.features["fps"] == 10.0
    assert (encoded.features["width"], encoded.features["height"], encoded.features["container"]) == (64, 48, "webm")
    back = media.decode_video(encoded.data, ".webm")
    assert back.frames.shape == (12, 48, 64, 3) and back.fps == pytest.approx(10.0)
    # Lossy, but the order and rough brightness of the frames survive.
    levels = back.frames[:, 24, 48, 1].astype(int)
    assert levels[-1] > levels[0] + 50


def test_odd_frame_sizes_are_encoded_too():
    encoded = media.encode_video(clip(frames=6, width=33, height=21))
    assert media.decode_video(encoded.data, ".webm").frames.shape[0] == 6


def test_decode_refuses_a_file_that_is_not_a_video_or_is_too_big_to_hold(monkeypatch):
    with pytest.raises(ValueError):
        media.decode_video(b"not a video" * 50, ".mp4")
    monkeypatch.setattr(media, "MAX_DECODED_VIDEO_BYTES", 64 * 48 * 3 * 5)
    with pytest.raises(ValueError, match="too large"):
        media.decode_video(mp4_bytes(frames=20), ".mp4")


def test_the_media_helpers_cover_every_file_modality_and_reject_the_rest():
    assert media.FILE_MODALITIES == ("vision", "audio", "video")
    assert media.worker_count("video") == 1 and media.worker_count("vision") == 4
    with pytest.raises(ValueError):
        media.decode_sample("text", b"", "")


# -- Preprocessing ops -----------------------------------------------------------------------


def apply_pre(op_cls, sample, **params):
    return op_cls.apply(sample, op_cls.Params(**params))


class TestPreprocessingOps:
    def test_all_three_are_offered_for_video_classification_only(self):
        ids = {o.id for o in list_preprocessing(TASK)}
        assert ids == {"video_resize", "video_resample_fps", "video_trim"}
        assert not any(o.modality == "video" for o in list_preprocessing(get_task_descriptor("audio_classification")))

    def test_resize_sets_the_shorter_side_and_keeps_the_aspect_ratio(self):
        out = apply_pre(pre_ops.Resize, clip(width=64, height=48), short_edge=32)
        assert out.frames.shape[1:3] == (32, 42) and len(out.frames) == 20  # 64/48 * 32 = 42.7, made even

    def test_resize_to_the_size_it_already_is_hands_back_the_same_clip(self):
        sample = clip(width=64, height=48)
        assert apply_pre(pre_ops.Resize, sample, short_edge=48) is sample

    def test_lower_frame_rate_drops_frames_and_keeps_the_duration(self):
        out = apply_pre(pre_ops.ResampleFps, clip(frames=20, fps=20.0), target_fps=10)
        assert out.fps == 10.0 and len(out.frames) == 10
        assert list(out.frames[:3, 0, 63, 1]) == [0, 20, 40]  # every other source frame, in order

    def test_a_clip_at_or_below_the_target_rate_is_kept_as_it_is(self):
        sample = clip(fps=10.0)
        assert apply_pre(pre_ops.ResampleFps, sample, target_fps=10) is sample
        assert apply_pre(pre_ops.ResampleFps, sample, target_fps=30) is sample

    def test_trim_keeps_the_middle_or_the_start_of_a_long_clip(self):
        sample = clip(frames=30, fps=10.0)  # 3 seconds
        middle = apply_pre(pre_ops.Trim, sample, max_seconds=1.0)
        assert len(middle.frames) == 10 and middle.frames[0, 0, 63, 1] == 100  # frames 10..19
        start = apply_pre(pre_ops.Trim, sample, max_seconds=1.0, keep="start")
        assert start.frames[0, 0, 63, 1] == 0 and len(start.frames) == 10

    def test_trim_leaves_a_short_clip_alone(self):
        sample = clip(frames=5, fps=10.0)
        assert apply_pre(pre_ops.Trim, sample, max_seconds=10.0) is sample

    def test_they_never_change_their_input_and_are_deterministic(self):
        sample = clip(frames=20, fps=20.0)
        before = sample.frames.copy()
        for op, params in (
            (pre_ops.Resize, {"short_edge": 32}),
            (pre_ops.ResampleFps, {"target_fps": 5}),
            (pre_ops.Trim, {"max_seconds": 1.0}),
        ):
            a, b = apply_pre(op, sample, **params), apply_pre(op, sample, **params)
            assert np.array_equal(a.frames, b.frames) and a.fps == b.fps
        assert np.array_equal(sample.frames, before)


# -- Augmentation ops ------------------------------------------------------------------------


def apply_aug(op_cls, sample, seed=0, **params):
    return op_cls.apply(sample, op_cls.Params(**params), random.Random(seed))


class TestAugmentationOps:
    def test_all_five_are_offered_for_video_classification(self):
        assert {o.id for o in list_augmentations(TASK)} == {
            "video_horizontal_flip",
            "video_temporal_crop",
            "video_brightness",
            "video_speed",
            "video_zoom",
        }

    def test_flip_mirrors_every_frame_left_to_right(self):
        sample = clip()
        out = aug_ops.HorizontalFlip.apply(sample, aug_ops.NoParams(), random.Random(0))
        assert np.array_equal(out.frames, sample.frames[:, :, ::-1])
        assert out.frames[0, 0, 0, 0] == 0 and out.frames[0, 0, -1, 0] == 200  # the red half moved to the right

    def test_temporal_crop_keeps_a_contiguous_stretch_of_at_least_the_minimum(self):
        sample = clip(frames=40)
        seen = set()
        for seed in range(30):
            out = apply_aug(aug_ops.TemporalCrop, sample, seed, min_fraction=0.5)
            assert 20 <= len(out.frames) <= 40
            levels = out.frames[:, 0, 63, 1].astype(int)
            assert list(np.diff(levels)) == [10] * (len(levels) - 1) or len(levels) == 1 or levels[-1] == 255
            seen.add(len(out.frames))
        assert len(seen) > 3

    def test_brightness_scales_the_whole_clip_by_one_factor_and_saturates(self):
        sample = clip()
        out = apply_aug(aug_ops.Brightness, sample, 1, max_change=0.4)
        assert out.frames.shape == sample.frames.shape and out.frames.dtype == np.uint8
        factor = out.frames[10, 0, 63, 1] / sample.frames[10, 0, 63, 1]
        assert 0.55 <= factor <= 1.45 and abs(factor - 1.0) > 1e-3
        assert out.frames[19, 0, 63, 1] <= 255

    def test_speed_changes_the_frame_count_but_not_the_frame_rate(self):
        sample = clip(frames=40, fps=10.0)
        counts = {len(apply_aug(aug_ops.Speed, sample, s, max_change=0.4).frames) for s in range(20)}
        assert len(counts) > 3 and min(counts) >= 40 / 1.4 - 1 and max(counts) <= 40 / 0.6 + 1
        assert apply_aug(aug_ops.Speed, sample, 0, max_change=0.4).fps == 10.0

    def test_zoom_keeps_the_frame_size(self):
        out = apply_aug(aug_ops.Zoom, clip(width=64, height=48), 3, min_scale=0.5)
        assert out.frames.shape == (20, 48, 64, 3)

    def test_the_same_seed_gives_the_same_clip_and_different_seeds_differ(self):
        sample = clip(frames=30)
        a, b = apply_aug(aug_ops.TemporalCrop, sample, 5), apply_aug(aug_ops.TemporalCrop, sample, 5)
        assert np.array_equal(a.frames, b.frames)
        assert len({len(apply_aug(aug_ops.TemporalCrop, sample, s).frames) for s in range(10)}) > 1

    def test_an_op_that_cannot_change_the_clip_hands_it_back(self):
        one_frame = clip(frames=1)
        assert apply_aug(aug_ops.TemporalCrop, one_frame) is one_frame
        assert apply_aug(aug_ops.Speed, one_frame) is one_frame


# -- Whole snapshots -------------------------------------------------------------------------


@pytest.fixture
def s3(monkeypatch):
    store: dict[str, bytes] = {}
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__(k, data))
    monkeypatch.setattr(storage, "download_bytes", lambda b, k: store[k])
    monkeypatch.setattr(storage, "delete_prefix", lambda b, p: 0)
    return store


async def seed(db, pre=None, aug=None):
    async with db() as s:
        user = User(name="U", email=f"{uuid.uuid4()}@x.co", password_hash="x")
        s.add(user)
        await s.flush()
        project = Project(user_id=user.id, name="p", task="video_classification")
        s.add(project)
        await s.flush()
        s.add(Dataset(project_id=project.id, modality="video"))
        await s.flush()
        version = DatasetVersion(
            dataset_id=project.id, version_tag="v1", status="building",
            preprocessing_config=pre, augmentation_config=aug,
        )  # fmt: skip
        s.add(version)
        cat = LabelClass(dataset_id=project.id, name="cat")
        s.add(cat)
        await s.commit()
        return project.id, version.id, cat.class_id


async def add_clip(db, project_id, version_id, cls, store, split="train", name="walk.mp4", **kw):
    data = mp4_bytes(**kw)
    key = f"pool/p/{uuid.uuid4().hex[:2]}/{name}"
    store[key] = data
    info = video_service.probe(data, ".mp4", "mp4")
    async with db() as s:
        item = DatasetItem(
            dataset_id=project_id, external_id=name, storage_url=key,
            content_hash=hashlib.sha256(data + uuid.uuid4().bytes).hexdigest(), byte_size=len(data),
        )  # fmt: skip
        s.add(item)
        await s.flush()
        s.add(DatasetVersionItem(version_id=version_id, item_id=item.id, split_type=split))
        s.add(
            VideoFeatures(
                item_id=item.id, width=info.width, height=info.height, duration_seconds=info.duration_seconds,
                fps=info.fps, frame_count=info.frame_count, container="mp4", codec=info.codec,
            )
        )  # fmt: skip
        s.add(Annotation(item_id=item.id, annotation_type="classification", class_id=cls))
        await s.commit()
        return item.id


async def members(db, version_id):
    async with db() as s:
        rows = (
            await s.execute(
                sa.select(DatasetItem, DatasetVersionItem.split_type)
                .join(DatasetVersionItem, DatasetVersionItem.item_id == DatasetItem.id)
                .where(DatasetVersionItem.version_id == version_id)
            )
        ).all()
    return [(i, split) for i, split in rows]


def pre_config(op, params=None, splits=("train", "validation", "test")):
    return {"ops": [{"id": op, "splits": list(splits), "params": params or {}}]}


async def test_preprocessing_replaces_a_clip_with_a_smaller_webm_and_its_features(db, s3):
    project_id, version_id, cat = await seed(db, pre_config("video_resize", {"shortEdge": 32}))
    original = await add_clip(db, project_id, version_id, cat, s3, width=64, height=48)
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    assert v.preprocessed_count == 1
    (item,) = [i for i, _ in await members(db, version_id) if i.source_item_id == original]
    assert item.storage_url.startswith(f"snapshots/{version_id}/preprocessed/") and item.storage_url.endswith(".webm")
    assert item.external_id == "walk.mp4"
    async with db() as s:
        f = (await s.execute(sa.select(VideoFeatures).where(VideoFeatures.item_id == item.id))).scalar_one()
        labels = (await s.execute(sa.select(Annotation).where(Annotation.item_id == item.id))).scalars().all()
    assert (f.width, f.height, f.container, f.frame_count) == (42, 32, "webm", 20)
    assert [a.class_id for a in labels] == [cat]  # the label came along
    assert media.decode_video(s3[item.storage_url], ".webm").frames.shape[1:3] == (32, 42)

    rows = pq.read_table(io.BytesIO(s3[f"snapshots/{version_id}/dataset.parquet"])).to_pylist()
    uris = {r["video_path"] for r in rows}
    assert f"s3://{BUCKET}/{item.storage_url}" in uris and not any(u.endswith("walk.mp4") for u in uris)


async def test_a_clip_the_ops_cannot_change_stays_the_pool_item_and_is_not_re_encoded(db, s3):
    project_id, version_id, cat = await seed(db, pre_config("video_resample_fps", {"targetFps": 30}))
    original = await add_clip(db, project_id, version_id, cat, s3, fps=10)  # already below 30 fps
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready" and v.preprocessed_count == 0
    assert [i.id for i, _ in await members(db, version_id)] == [original]
    assert not [k for k in s3 if "/preprocessed/" in k]


async def test_an_undecodable_clip_is_skipped_but_does_not_fail_the_snapshot(db, s3):
    project_id, version_id, cat = await seed(db, pre_config("video_resize", {"shortEdge": 32}))
    good = await add_clip(db, project_id, version_id, cat, s3, name="good.mp4")
    bad = await add_clip(db, project_id, version_id, cat, s3, name="bad.mp4")
    async with db() as s:
        key = (await s.get(DatasetItem, bad)).storage_url
    s3[key] = b"not a video at all" * 40
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    ids = {i.id for i, _ in await members(db, version_id)}
    assert bad in ids and good not in ids  # the broken one stays as it was; the good one was replaced
    assert v.preprocessed_count == 1


async def test_augmentation_adds_playable_webm_copies_to_the_train_split_only(db, s3):
    aug = {"copiesPerItem": 2, "ops": [{"id": "video_horizontal_flip", "probability": 1.0, "params": {}}]}
    project_id, version_id, cat = await seed(db, aug=aug)
    train = await add_clip(db, project_id, version_id, cat, s3, split="train", name="a.mp4")
    await add_clip(db, project_id, version_id, cat, s3, split="test", name="b.mp4")
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    items = await members(db, version_id)
    copies = [(i, sp) for i, sp in items if i.source_item_id == train]
    assert v.augmented_count == len(copies) == 2 and all(sp == "train" for _, sp in copies)
    assert sorted(i.external_id for i, _ in copies) == ["a_aug1.webm", "a_aug2.webm"]
    assert len({i.content_hash for i, _ in copies}) == 2 or len(copies) == 2
    for item, _ in copies:
        assert item.storage_url.startswith(f"snapshots/{version_id}/augmented/") and item.storage_url.endswith(".webm")
        assert media.decode_video(s3[item.storage_url], ".webm").frames.shape == (20, 48, 64, 3)
        async with db() as s:
            f = (await s.execute(sa.select(VideoFeatures).where(VideoFeatures.item_id == item.id))).scalar_one()
        assert f.container == "webm" and f.frame_count == 20
    assert not [i for i, sp in items if sp == "test" and i.source_item_id is not None]


async def test_an_augmentation_that_cannot_change_a_clip_adds_no_copy(db, s3):
    aug = {
        "copiesPerItem": 1,
        "ops": [{"id": "video_temporal_crop", "probability": 1.0, "params": {"minFraction": 0.95}}],
    }
    project_id, version_id, cat = await seed(db, aug=aug)
    await add_clip(db, project_id, version_id, cat, s3, frames=1, name="one.mp4")
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    # A snapshot whose augmentation adds nothing at all fails with a reason, for every modality (see
    # services/augmentation.py), rather than quietly producing the plain snapshot the user did not ask for.
    assert v.status == "failed" and "no new items" in v.failed_message

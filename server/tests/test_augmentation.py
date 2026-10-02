"""Augmentation plugins: discovery, parameter introspection, config validation, and every op's output."""

import io
import random

import numpy as np
import pytest
from PIL import Image

from theseus.augmentation import media
from theseus.augmentation.base import Augmentation, NoParams
from theseus.augmentation.config import AugmentationConfig, AugmentationOpConfig
from theseus.augmentation.media import AudioClip
from theseus.augmentation.pipeline import augment, build_plan, seed_for
from theseus.augmentation.registry import (
    AugmentationConfigError,
    describe,
    get_augmentation,
    list_augmentations,
    param_specs,
    validate_config,
)
from theseus.services.task_registry import get_task_descriptor

IMAGE_TASK = get_task_descriptor("image_classification")
TEXT_TASK = get_task_descriptor("text_classification")
AUDIO_TASK = get_task_descriptor("audio_classification")
TABULAR_TASK = get_task_descriptor("tabular_classification")
REGRESSION_TASK = get_task_descriptor("tabular_regression")


def cfg(*ops, copies=1):
    return AugmentationConfig(copies_per_item=copies, ops=[AugmentationOpConfig(id=i, **kw) for i, kw in ops])


def sample_for(modality):
    if modality == "vision":
        rng = np.random.default_rng(1)
        return Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    if modality == "text":
        return "the quick brown fox jumps over the lazy dog again and again"
    if modality == "audio":
        t = np.linspace(0, 1, 8000, dtype=np.float32)
        return AudioClip((0.5 * np.sin(2 * np.pi * 220 * t))[None, :], 8000)
    return {"age": 30, "income": 52000.5, "score": 7, "name": "x", "flag": True}


TABULAR_ROWS = [{"age": a, "income": a * 1000.0, "score": a % 5, "name": "x", "flag": True} for a in range(20, 60, 4)]


def run_op(op_id, seed=0, params=None):
    op = get_augmentation(op_id)
    modality = op.modality
    originals = TABULAR_ROWS if modality == "tabular" else []
    plan = build_plan(cfg((op_id, {"probability": 1.0, "params": params or {}})), originals)
    sample = sample_for(modality)
    return sample, augment(sample, plan, random.Random(seed))[0]


# -- Discovery -------------------------------------------------------------------------------


def test_every_shipped_op_is_discovered_and_grouped_by_modality():
    ids = [(op.modality, op.id) for op in list_augmentations()]
    assert [m for m, _ in ids] == sorted((m for m, _ in ids), key=("vision", "text", "audio", "tabular").index)
    assert {"image_rotate", "text_word_deletion", "audio_noise", "tabular_gaussian_noise"} <= {i for _, i in ids}
    assert len({i for _, i in ids}) == len(ids)


def test_ops_are_offered_only_for_label_preserving_tasks_of_their_own_modality():
    assert {op.modality for op in list_augmentations(IMAGE_TASK)} == {"vision"}
    assert {op.modality for op in list_augmentations(TEXT_TASK)} == {"text"}
    assert {op.modality for op in list_augmentations(AUDIO_TASK)} == {"audio"}
    assert {op.modality for op in list_augmentations(TABULAR_TASK)} == {"tabular"}
    assert list_augmentations(REGRESSION_TASK)  # regression keeps the target, so tabular ops apply
    # spatial / sequence / generative / planned tasks would need labels transformed in step: none offered
    for task_id in ("token_classification", "text_generation", "image_captioning", "object_detection"):
        assert list_augmentations(get_task_descriptor(task_id)) == [], task_id


def test_a_new_op_class_is_picked_up_with_no_other_change():
    from theseus.augmentation.base import _registry

    class Shout(Augmentation):
        id = "test_only_shout"
        label = "Shout"
        modality = "text"

        @classmethod
        def apply(cls, sample, params, rng, state=None):
            return sample.upper()

    try:
        assert get_augmentation("test_only_shout") is Shout
        assert Shout in list_augmentations(TEXT_TASK) and Shout not in list_augmentations(IMAGE_TASK)
        assert describe(Shout).params == []
    finally:
        _registry.unregister("test_only_shout")


def test_a_duplicate_op_id_fails_loudly():
    list_augmentations()
    with pytest.raises(ValueError, match="Duplicate augmentation id 'image_rotate'"):

        class Clash(Augmentation):
            id = "image_rotate"
            label = "Clash"
            modality = "vision"

            @classmethod
            def apply(cls, sample, params, rng, state=None):
                return sample


# -- Parameter introspection -----------------------------------------------------------------


def test_param_specs_flatten_a_params_model_for_the_web_form():
    (spec,) = param_specs(get_augmentation("image_rotate"))
    assert (spec.name, spec.type, spec.default, spec.min, spec.max) == ("maxDegrees", "float", 15.0, 1.0, 45.0)
    assert spec.label == "Max rotation (degrees)" and spec.step == 1
    assert param_specs(get_augmentation("image_horizontal_flip")) == []
    (swaps,) = param_specs(get_augmentation("text_word_swap"))
    assert (swaps.type, swaps.min, swaps.max, swaps.step) == ("int", 1.0, 10.0, 1.0)


def test_every_shipped_op_can_be_described():
    for op in list_augmentations():
        info = describe(op)
        assert info.id == op.id and info.label and info.description, op.id
        assert all(p.name and p.label and p.default is not None for p in info.params), op.id
        assert all(p.min is not None and p.max is not None for p in info.params if p.type in ("int", "float")), op.id


# -- Config validation -----------------------------------------------------------------------


def test_validate_config_fills_defaults_so_the_snapshot_records_what_was_applied():
    out = validate_config(
        IMAGE_TASK, cfg(("image_rotate", {"params": {"maxDegrees": 20}}), ("image_horizontal_flip", {}))
    )
    assert out.ops[0].params == {"maxDegrees": 20.0} and out.ops[1].params == {}
    assert validate_config(IMAGE_TASK, cfg(("image_rotate", {}))).ops[0].params == {"maxDegrees": 15.0}


@pytest.mark.parametrize(
    ("task", "ops", "message"),
    [
        (IMAGE_TASK, [("nope", {})], "Unknown augmentation 'nope'"),
        (IMAGE_TASK, [("text_typos", {})], "not available for Image Classification"),
        (
            IMAGE_TASK,
            [("image_rotate", {"params": {"maxDegrees": 999}})],
            "Invalid parameter 'maxDegrees' for 'image_rotate'",
        ),
        (IMAGE_TASK, [("image_rotate", {"params": {"bogus": 1}})], "Invalid parameter 'bogus'"),
        (IMAGE_TASK, [("image_rotate", {}), ("image_rotate", {})], "listed more than once"),
        (get_task_descriptor("object_detection"), [("image_rotate", {})], "not available"),
    ],
)
def test_validate_config_rejects_bad_requests_with_a_user_safe_message(task, ops, message):
    with pytest.raises(AugmentationConfigError, match=message):
        validate_config(task, cfg(*ops))


def test_config_bounds_are_enforced_by_the_model():
    with pytest.raises(ValueError):
        cfg(("image_rotate", {}), copies=11)
    with pytest.raises(ValueError):
        AugmentationConfig(copies_per_item=1, ops=[])
    with pytest.raises(ValueError):
        AugmentationOpConfig(id="x", probability=1.5)


# -- Every op produces a valid, different sample ---------------------------------------------


@pytest.mark.parametrize("op", [o.id for o in list_augmentations() if o.modality == "vision"])
def test_image_ops_return_a_valid_image_of_the_same_mode_and_size(op):
    src, out = run_op(op)
    assert isinstance(out, Image.Image) and out.mode == src.mode and out.size == src.size
    assert out.tobytes() != src.tobytes()  # the source is random noise, so every op visibly changes it


@pytest.mark.parametrize("op", [o.id for o in list_augmentations() if o.modality == "text"])
def test_text_ops_return_a_changed_nonempty_string(op):
    # Rate-based ops can legitimately leave a short text untouched at their defaults (the snapshot build
    # skips such copies), so use strong settings to check the op itself.
    strong = {"text_word_deletion": {"deleteRate": 0.5}, "text_typos": {"typoRate": 0.1}}
    src, out = run_op(op, params=strong.get(op))
    assert isinstance(out, str) and out.strip() and out != src


@pytest.mark.parametrize("op", [o.id for o in list_augmentations() if o.modality == "audio"])
def test_audio_ops_return_a_valid_clip_within_range(op):
    src, out = run_op(op)
    assert isinstance(out, AudioClip) and out.sample_rate == src.sample_rate and out.samples.shape[0] == 1
    assert out.samples.dtype == np.float32 and np.abs(out.samples).max() <= 1.0 and out.samples.shape[1] > 0
    assert not np.array_equal(out.samples, src.samples)


@pytest.mark.parametrize("op", [o.id for o in list_augmentations() if o.modality == "tabular"])
def test_tabular_ops_keep_the_schema_and_never_touch_non_numeric_cells(op):
    src, out = run_op(op, params={"dropoutRate": 0.5} if op == "tabular_feature_dropout" else None)
    assert set(out) == set(src) and out["name"] == "x" and out["flag"] is True
    assert out != src
    assert isinstance(out["score"], int)  # an integer column stays integer-valued


# -- Determinism and the pipeline ------------------------------------------------------------


def test_the_same_seed_reproduces_the_same_copy_and_different_seeds_differ():
    a1, b1 = run_op("text_word_swap", seed=1)[1], run_op("text_word_swap", seed=1)[1]
    assert a1 == b1
    assert len({run_op("image_rotate", seed=s)[1].tobytes() for s in range(5)}) > 1
    assert seed_for("v", "i", 0) == seed_for("v", "i", 0) != seed_for("v", "i", 1) != seed_for("w", "i", 1)


def test_when_no_op_fires_one_is_forced_so_a_copy_is_never_a_plain_duplicate():
    plan = build_plan(cfg(("text_word_swap", {"probability": 0.0}), ("text_typos", {"probability": 0.0})), [])
    out, applied = augment(sample_for("text"), plan, random.Random(3))
    assert len(applied) == 1 and applied[0]["id"] in {"text_word_swap", "text_typos"}


def test_applied_ops_are_recorded_with_their_params():
    plan = build_plan(cfg(("image_rotate", {"probability": 1.0, "params": {"maxDegrees": 30}})), [])
    _, applied = augment(sample_for("vision"), plan, random.Random(0))
    assert applied == [{"id": "image_rotate", "params": {"maxDegrees": 30.0}}]


def test_ops_do_not_mutate_their_input():
    img = sample_for("vision")
    before = img.tobytes()
    plan = build_plan(
        cfg(("image_gaussian_noise", {"probability": 1.0}), ("image_horizontal_flip", {"probability": 1.0})), []
    )
    augment(img, plan, random.Random(0))
    assert img.tobytes() == before
    row = dict(TABULAR_ROWS[0])
    plan = build_plan(cfg(("tabular_gaussian_noise", {"probability": 1.0})), TABULAR_ROWS)
    augment(row, plan, random.Random(0))
    assert row == TABULAR_ROWS[0]


def test_word_deletion_always_keeps_at_least_one_word():
    plan = build_plan(cfg(("text_word_deletion", {"probability": 1.0, "params": {"deleteRate": 0.5}})), [])
    for seed in range(30):
        assert augment("one two", plan, random.Random(seed))[0].strip()


# -- Media codecs ----------------------------------------------------------------------------


def png_bytes(mode="RGB", size=(16, 12)):
    buf = io.BytesIO()
    Image.new(mode, size, "red" if mode == "RGB" else 128).save(buf, format="PNG")
    return buf.getvalue()


def test_images_round_trip_through_the_codec_keeping_jpeg_and_normalising_the_rest_to_png():
    img = media.decode_image(png_bytes())
    jpg = media.encode_image(img, ".jpg")
    assert (jpg.ext, jpg.content_type, jpg.features["image_format"]) == (".jpg", "image/jpeg", "jpeg")
    assert Image.open(io.BytesIO(jpg.data)).format == "JPEG"
    other = media.encode_image(img, ".webp")
    assert (other.ext, other.features) == (".png", {"width": 16, "height": 12, "channels": 3, "image_format": "png"})


def test_palette_and_alpha_images_are_flattened_to_rgb_before_augmenting():
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(buf, format="PNG")
    assert media.decode_image(buf.getvalue()).mode == "RGB"
    assert media.decode_image(png_bytes("L")).mode == "L"


def test_wav_audio_round_trips_and_reports_its_features():
    clip = sample_for("audio")
    enc = media.encode_audio(clip)
    assert (enc.ext, enc.content_type) == (".wav", "audio/wav")
    assert enc.features == {"duration_seconds": 1.0, "sample_rate_hz": 8000, "channels": 1, "audio_codec": "wav"}
    back = media.decode_audio(enc.data, ".wav")
    assert back.sample_rate == 8000 and back.samples.shape == clip.samples.shape
    assert np.allclose(back.samples, clip.samples, atol=1e-3)


def test_stereo_audio_keeps_its_channels():
    clip = AudioClip(np.stack([sample_for("audio").samples[0], -sample_for("audio").samples[0]]), 8000)
    back = media.decode_audio(media.encode_audio(clip).data, ".wav")
    assert back.samples.shape == (2, 8000) and back.samples[0, 100] == pytest.approx(-back.samples[1, 100], abs=1e-3)


def test_noop_params_model_accepts_only_an_empty_dict():
    assert NoParams.model_validate({}) == NoParams()
    with pytest.raises(ValueError):
        NoParams.model_validate({"x": 1})

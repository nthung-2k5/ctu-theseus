"""Preprocessing plugins: discovery, parameter introspection, config validation, and every op's output."""

import numpy as np
import pytest
from PIL import Image

from theseus.augmentation.media import AudioClip
from theseus.preprocessing.base import NoParams, Preprocessing
from theseus.preprocessing.config import PreprocessingConfig, PreprocessingOpConfig
from theseus.preprocessing.pipeline import build_plan, ops_for_split, preprocess
from theseus.preprocessing.registry import (
    PreprocessingConfigError,
    describe,
    get_preprocessing,
    list_preprocessing,
    param_specs,
    validate_config,
)
from theseus.services.task_registry import get_task_descriptor

IMAGE_TASK = get_task_descriptor("image_classification")
TEXT_TASK = get_task_descriptor("text_classification")
AUDIO_TASK = get_task_descriptor("audio_classification")
TABULAR_TASK = get_task_descriptor("tabular_classification")
REGRESSION_TASK = get_task_descriptor("tabular_regression")


def cfg(*ops):
    return PreprocessingConfig(ops=[PreprocessingOpConfig(id=i, **kw) for i, kw in ops])


def sample_for(modality):
    if modality == "vision":
        rng = np.random.default_rng(1)
        return Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    if modality == "text":
        return "The Quick   Brown Fox"
    if modality == "audio":
        t = np.linspace(0, 1, 8000, dtype=np.float32)
        left = 0.5 * np.sin(2 * np.pi * 220 * t)
        right = 0.2 * np.cos(2 * np.pi * 220 * t)
        return AudioClip(np.stack([left, right]).astype(np.float32), 8000)
    return {"age": 30, "income": 52000.5, "score": 7, "name": "x", "flag": True}


TABULAR_ROWS = [{"age": a, "income": a * 1000.0, "score": a % 5, "name": "x", "flag": True} for a in range(20, 60, 4)]


def run_op(op_id, params=None, train_samples=None):
    op = get_preprocessing(op_id)
    modality = op.modality
    samples = train_samples if train_samples is not None else (TABULAR_ROWS if modality == "tabular" else [])
    plan = build_plan(cfg((op_id, {"params": params or {}})), samples)
    sample = sample_for(modality)
    return sample, preprocess(sample, plan)[0]


# -- Discovery -------------------------------------------------------------------------------


def test_every_shipped_op_is_discovered_and_grouped_by_modality():
    ids = [(op.modality, op.id) for op in list_preprocessing()]
    assert [m for m, _ in ids] == sorted((m for m, _ in ids), key=("vision", "text", "audio", "tabular").index)
    assert {"image_resize", "text_lowercase", "audio_resample", "tabular_standardize"} <= {i for _, i in ids}
    assert len({i for _, i in ids}) == len(ids)


def test_ops_are_offered_only_for_label_preserving_tasks_of_their_own_modality():
    assert {op.modality for op in list_preprocessing(IMAGE_TASK)} == {"vision"}
    assert {op.modality for op in list_preprocessing(TEXT_TASK)} == {"text"}
    assert {op.modality for op in list_preprocessing(AUDIO_TASK)} == {"audio"}
    assert {op.modality for op in list_preprocessing(TABULAR_TASK)} == {"tabular"}
    assert list_preprocessing(REGRESSION_TASK)  # regression keeps the target, so tabular ops apply
    for task_id in ("token_classification", "text_generation", "image_captioning", "object_detection"):
        assert list_preprocessing(get_task_descriptor(task_id)) == [], task_id


def test_a_new_op_class_is_picked_up_with_no_other_change():
    from theseus.preprocessing.base import _registry

    class Shout(Preprocessing):
        id = "test_only_shout"
        label = "Shout"
        modality = "text"

        @classmethod
        def apply(cls, sample, params, state=None):
            return sample.upper()

    try:
        assert get_preprocessing("test_only_shout") is Shout
        assert Shout in list_preprocessing(TEXT_TASK) and Shout not in list_preprocessing(IMAGE_TASK)
        assert describe(Shout).params == []
    finally:
        _registry.unregister("test_only_shout")


def test_a_duplicate_op_id_fails_loudly():
    list_preprocessing()
    with pytest.raises(ValueError, match="Duplicate preprocessing id 'image_resize'"):

        class Clash(Preprocessing):
            id = "image_resize"
            label = "Clash"
            modality = "vision"

            @classmethod
            def apply(cls, sample, params, state=None):
                return sample


# -- Parameter introspection -----------------------------------------------------------------


def test_param_specs_flatten_a_params_model_for_the_web_form():
    specs = param_specs(get_preprocessing("image_resize"))
    by_name = {s.name: s for s in specs}
    assert (by_name["width"].type, by_name["width"].default) == ("int", 224)
    assert by_name["mode"].type == "choice" and set(by_name["mode"].choices) == {"stretch", "fit"}
    assert param_specs(get_preprocessing("image_grayscale")) == []


def test_every_shipped_op_can_be_described():
    for op in list_preprocessing():
        info = describe(op)
        assert info.id == op.id and info.label and info.description, op.id
        assert all(p.name and p.label and p.default is not None for p in info.params), op.id


# -- Config validation -----------------------------------------------------------------------


def test_validate_config_fills_defaults_so_the_snapshot_records_what_was_applied():
    out = validate_config(IMAGE_TASK, cfg(("image_resize", {"params": {"width": 100, "height": 100}})))
    assert out.ops[0].params == {"width": 100.0, "height": 100.0, "mode": "stretch"}
    assert out.ops[0].splits == ["train", "validation", "test"]


def test_validate_config_keeps_explicit_split_selection():
    out = validate_config(IMAGE_TASK, cfg(("image_resize", {"splits": ["train", "test"]})))
    assert out.ops[0].splits == ["train", "test"]


@pytest.mark.parametrize(
    ("task", "ops", "message"),
    [
        (IMAGE_TASK, [("nope", {})], "Unknown preprocessing op 'nope'"),
        (IMAGE_TASK, [("text_lowercase", {})], "not available for Image Classification"),
        (
            IMAGE_TASK,
            [("image_resize", {"params": {"width": 99999}})],
            "Invalid parameter 'width' for 'image_resize'",
        ),
        (IMAGE_TASK, [("image_resize", {"params": {"bogus": 1}})], "Invalid parameter 'bogus'"),
        (IMAGE_TASK, [("image_resize", {}), ("image_resize", {})], "listed more than once"),
        (get_task_descriptor("object_detection"), [("image_resize", {})], "not available"),
    ],
)
def test_validate_config_rejects_bad_requests_with_a_user_safe_message(task, ops, message):
    with pytest.raises(PreprocessingConfigError, match=message):
        validate_config(task, cfg(*ops))


def test_config_bounds_are_enforced_by_the_model():
    with pytest.raises(ValueError):
        PreprocessingConfig(ops=[])
    with pytest.raises(ValueError):
        PreprocessingOpConfig(id="x", splits=[])
    with pytest.raises(ValueError):
        PreprocessingOpConfig(id="x", splits=["train", "train"])


# -- The pipeline: split scoping ---------------------------------------------------------------


def test_ops_for_split_only_returns_ops_scoped_to_that_split():
    plan = build_plan(
        cfg(("image_resize", {"splits": ["train", "validation"]}), ("image_grayscale", {"splits": ["test"]})), []
    )
    assert [c.op.id for c in ops_for_split(plan, "train")] == ["image_resize"]
    assert [c.op.id for c in ops_for_split(plan, "test")] == ["image_grayscale"]
    assert ops_for_split(plan, "validation")[0].op.id == "image_resize"


def test_applied_ops_are_recorded_with_their_params():
    plan = build_plan(cfg(("image_resize", {"params": {"width": 50, "height": 60}})), [])
    _, applied = preprocess(sample_for("vision"), plan)
    assert applied == [{"id": "image_resize", "params": {"width": 50.0, "height": 60.0, "mode": "stretch"}}]


# -- Every op is deterministic and produces a valid sample ------------------------------------


@pytest.mark.parametrize("op", [o.id for o in list_preprocessing() if o.modality == "vision"])
def test_image_ops_are_deterministic_and_return_a_valid_image(op):
    src, out1 = run_op(op)
    _, out2 = run_op(op)
    assert isinstance(out1, Image.Image) and out1.tobytes() == out2.tobytes()  # same input, same output


def test_image_resize_stretches_to_the_target_size():
    src, out = run_op("image_resize", params={"width": 50, "height": 40, "mode": "stretch"})
    assert out.size == (50, 40)


def test_image_resize_fit_mode_keeps_aspect_and_pads():
    src, out = run_op("image_resize", params={"width": 64, "height": 64, "mode": "fit"})
    assert out.size == (64, 64) and out.mode == src.mode


def test_image_grayscale_converts_to_single_channel():
    _, out = run_op("image_grayscale")
    assert out.mode == "L"


@pytest.mark.parametrize("op", [o.id for o in list_preprocessing() if o.modality == "text"])
def test_text_ops_are_deterministic(op):
    _, out1 = run_op(op)
    _, out2 = run_op(op)
    assert isinstance(out1, str) and out1 == out2


def test_text_lowercase_lowercases_everything():
    _, out = run_op("text_lowercase")
    assert out == "the quick   brown fox"


def test_normalize_whitespace_collapses_and_trims():
    sample, out = run_op("text_normalize_whitespace")
    assert out == "The Quick Brown Fox"


@pytest.mark.parametrize("op", [o.id for o in list_preprocessing() if o.modality == "audio"])
def test_audio_ops_are_deterministic_and_return_a_valid_clip(op):
    src, out1 = run_op(op)
    _, out2 = run_op(op)
    assert isinstance(out1, AudioClip) and np.array_equal(out1.samples, out2.samples)


def test_audio_resample_changes_the_sample_rate_and_frame_count():
    src, out = run_op("audio_resample", params={"sampleRateHz": 4000})
    assert out.sample_rate == 4000 and out.samples.shape[1] == pytest.approx(src.samples.shape[1] / 2, rel=0.01)


def test_audio_resample_is_a_noop_at_the_same_rate():
    src, out = run_op("audio_resample", params={"sampleRateHz": 8000})
    assert out.sample_rate == src.sample_rate and out.samples.shape == src.samples.shape


def test_audio_to_mono_averages_channels():
    src, out = run_op("audio_to_mono")
    assert out.samples.shape[0] == 1
    assert np.allclose(out.samples[0], np.mean(src.samples, axis=0))


@pytest.mark.parametrize("op", [o.id for o in list_preprocessing() if o.modality == "tabular"])
def test_tabular_ops_keep_the_schema_and_never_touch_non_numeric_cells(op):
    src, out = run_op(op)
    assert set(out) == set(src) and out["name"] == "x" and out["flag"] is True


def test_tabular_standardize_fits_mean_and_std_from_train_only():
    train_ages = [0, 10, 20, 30, 40]  # mean 20, population std ~14.142; the sample's age (30) is not in it
    train = [{"age": a} for a in train_ages]
    sample, out = run_op("tabular_standardize", train_samples=train)
    assert out["age"] == pytest.approx((sample["age"] - np.mean(train_ages)) / np.std(train_ages), rel=1e-6)


def test_tabular_min_max_scales_into_0_1_using_train_bounds():
    train = [{"age": a} for a in (0, 100)]
    sample, out = run_op("tabular_min_max", train_samples=train)
    assert out["age"] == pytest.approx((sample["age"] - 0) / (100 - 0))


def test_a_constant_column_is_left_unscaled_to_avoid_dividing_by_zero():
    train = [{"age": 50}, {"age": 50}, {"age": 50}]  # constant, and different from the sample's age (30)
    sample, out = run_op("tabular_standardize", train_samples=train)
    assert out["age"] == sample["age"] == 30  # std is 0: standardizing would divide by zero, so left as-is
    sample, out = run_op("tabular_min_max", train_samples=train)
    assert out["age"] == sample["age"] == 30  # min == max: min-max would divide by zero, so left as-is


# -- Ops do not mutate their input -------------------------------------------------------------


def test_ops_do_not_mutate_their_input():
    img = sample_for("vision")
    before = img.tobytes()
    plan = build_plan(cfg(("image_grayscale", {})), [])
    preprocess(img, plan)
    assert img.tobytes() == before

    row = dict(TABULAR_ROWS[0])
    plan = build_plan(cfg(("tabular_standardize", {})), TABULAR_ROWS)
    preprocess(row, plan)
    assert row == TABULAR_ROWS[0]


def test_noop_params_model_accepts_only_an_empty_dict():
    assert NoParams.model_validate({}) == NoParams()
    with pytest.raises(ValueError):
        NoParams.model_validate({"x": 1})

import io
import json
import uuid

import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa

from theseus.db.models import (
    Annotation,
    Dataset,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    Project,
    TabularFeatures,
    TextFeatures,
    User,
)
from theseus.services import snapshot as snap
from theseus.services import storage
from theseus.services.task_registry import ColumnSpec, get_task_descriptor

CLASS_ID = uuid.uuid4()
ITEM_ID = uuid.uuid4()
CLASS_NAMES = {CLASS_ID: "cat"}


def member(**kw) -> snap.SnapshotMember:
    base = dict(
        item_id=ITEM_ID,
        split_type="train",
        storage_url="pool/abc.png",
        raw_text="hello world",
        features_json={"age": 42},
        annotations=[snap.AnnotationView(CLASS_ID, "classification")],
    )
    return snap.SnapshotMember(**{**base, **kw})


def col(name: str, kind: str) -> ColumnSpec:
    return ColumnSpec(name=name, kind=kind)


class TestResolveColumnValue:
    def test_item_id_resolves_to_the_pool_item_id(self):
        assert snap.resolve_column_value(col("_theseus_item_id", "item_id"), member(), CLASS_NAMES) == str(ITEM_ID)

    def test_other_kinds(self):
        m = member()
        assert snap.resolve_column_value(col("split", "split"), m, CLASS_NAMES) == "train"
        assert snap.resolve_column_value(col("idx", "split_index"), m, CLASS_NAMES) == 0
        assert snap.resolve_column_value(col("class", "label"), m, CLASS_NAMES) == "cat"
        assert snap.resolve_column_value(col("text", "inline_text"), m, CLASS_NAMES) == "hello world"
        assert snap.resolve_column_value(col("age", "scalar"), m, CLASS_NAMES) == 42
        assert (
            snap.resolve_column_value(col("image_path", "storage_uri"), m, CLASS_NAMES)
            == "s3://theseus-datasets/pool/abc.png"
        )

    def test_split_index_follows_ludwig_fixed_split_convention(self):
        vals = [
            snap.resolve_column_value(col("i", "split_index"), member(split_type=s), {})
            for s in ("train", "validation", "test")
        ]
        assert vals == [0, 1, 2]

    def test_text_sequence_label_reads_the_text_sequence_annotation_not_classification(self):
        captioned = member(annotations=[snap.AnnotationView(None, "text_sequence", None, "a cat on a mat")])
        c = col("caption", "text_sequence_label")
        assert snap.resolve_column_value(c, captioned, CLASS_NAMES) == "a cat on a mat"
        assert snap.resolve_column_value(c, member(), CLASS_NAMES) is None

    def test_regression_label_falls_back_to_label_structured_value(self):
        m = member(annotations=[snap.AnnotationView(None, "classification", {"value": 3.5})])
        assert snap.resolve_column_value(col("target", "label"), m, {}) == 3.5

    def test_missing_values_are_none(self):
        m = member(storage_url=None, raw_text=None, features_json=None, annotations=[])
        for kind in ("storage_uri", "inline_text", "label", "scalar", "text_sequence_label"):
            assert snap.resolve_column_value(col("c", kind), m, {}) is None


class TestColumnsAndParquet:
    def test_tabular_scalar_columns_are_the_union_of_feature_keys_in_first_seen_order(self):
        task = get_task_descriptor("tabular_classification")
        members = [member(features_json={"age": 1}), member(features_json={"income": 2, "age": 3})]
        names = [c.name for c in snap.derive_columns(task, members)]
        assert names == ["class", "split", "age", "income", "_ludwig_split_idx", "_theseus_item_id"]

    def test_non_tabular_tasks_get_only_synthetic_extras(self):
        task = get_task_descriptor("image_classification")
        assert [c.name for c in snap.derive_columns(task, [member()])][-2:] == ["_ludwig_split_idx", "_theseus_item_id"]

    def test_class_counts_only_for_classification_tasks(self):
        rows = [{"class": "cat"}, {"class": "cat"}, {"class": "dog"}, {"class": None}]
        cols = [col("class", "label")]
        assert snap.class_counts(get_task_descriptor("image_classification"), cols, rows) == {"cat": 2, "dog": 1}
        assert (
            snap.class_counts(get_task_descriptor("tabular_regression"), [col("target", "label")], [{"target": 1.0}])
            is None
        )

    def test_parquet_has_task_column_order_and_types(self):
        task = get_task_descriptor("tabular_regression")
        members = [
            member(features_json={"age": 40}, annotations=[snap.AnnotationView(None, "classification", {"value": 2.5})])
        ]
        columns = snap.derive_columns(task, members)
        rows = snap.build_rows(columns, members, {})
        table = pq.read_table(io.BytesIO(snap.to_parquet_bytes(task, columns, rows)))
        assert table.column_names == ["target", "split", "age", "_ludwig_split_idx", "_theseus_item_id"]
        types = {f.name: str(f.type) for f in table.schema}
        assert types == {
            "target": "double",
            "split": "string",
            "age": "double",
            "_ludwig_split_idx": "int32",
            "_theseus_item_id": "string",
        }
        assert table.to_pylist()[0]["target"] == 2.5

    def test_non_numeric_cell_in_a_numeric_column_becomes_null_instead_of_failing(self):
        task = get_task_descriptor("tabular_regression")
        members = [member(features_json={"age": "n/a"}, annotations=[])]
        columns = snap.derive_columns(task, members)
        table = pq.read_table(io.BytesIO(snap.to_parquet_bytes(task, columns, snap.build_rows(columns, members, {}))))
        assert table.to_pylist()[0]["age"] is None

    def test_manifest_shape_matches_what_the_compiler_reads_back(self):
        columns = [col("class", "label")]
        m = snap.build_manifest([member()], ["cat", "dog"], {"cat": 1}, columns)
        assert (m["itemCount"], m["classCount"], m["classes"], m["classCounts"]) == (1, 2, ["cat", "dog"], {"cat": 1})
        assert m["columns"] == [{"name": "class", "kind": "label"}]
        assert "classCounts" not in snap.build_manifest([], [], None, columns)


# -- End to end (real Postgres, stubbed S3) ---------------------------------------------------


@pytest.fixture
def fake_s3(monkeypatch):
    store: dict[str, bytes] = {}
    monkeypatch.setattr(
        storage, "upload_bytes", lambda bucket, key, data, content_type=None: store.__setitem__(key, data)
    )
    monkeypatch.setattr(storage, "download_bytes", lambda bucket, key: store[key])
    return store


async def _seed(s, task: str, modality: str):
    user = User(name="U", email=f"{uuid.uuid4()}@x.co", password_hash="x")
    s.add(user)
    await s.flush()
    project = Project(user_id=user.id, name="p", task=task)
    s.add(project)
    await s.flush()
    s.add(Dataset(project_id=project.id, modality=modality))
    await s.flush()
    version = DatasetVersion(dataset_id=project.id, version_tag="v1", status="building")
    s.add(version)
    await s.flush()
    return project, version


async def test_build_snapshot_writes_parquet_and_manifest_and_marks_ready(db, fake_s3):
    async with db() as s:
        project, version = await _seed(s, "text_classification", "text")
        cats = LabelClass(dataset_id=project.id, name="cat")
        dogs = LabelClass(dataset_id=project.id, name="dog")
        s.add_all([cats, dogs])
        await s.flush()
        items = []
        for text, cls, split in [("meow", cats, "train"), ("purr", cats, "validation"), ("woof", dogs, "test")]:
            item = DatasetItem(dataset_id=project.id, content_hash=uuid.uuid4().hex + uuid.uuid4().hex)
            s.add(item)
            await s.flush()
            s.add_all(
                [
                    TextFeatures(item_id=item.id, raw_text=text),
                    DatasetVersionItem(version_id=version.id, item_id=item.id, split_type=split),
                    Annotation(item_id=item.id, annotation_type="classification", class_id=cls.class_id),
                ]
            )
            items.append(item)
        await s.commit()
        version_id = version.id

    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
        assert (v.status, v.item_count, v.class_count) == ("ready", 3, 2)
        assert v.parquet_key == f"snapshots/{version_id}/dataset.parquet" and v.built_at is not None

    table = pq.read_table(io.BytesIO(fake_s3[f"snapshots/{version_id}/dataset.parquet"]))
    rows = {r["text"]: r for r in table.to_pylist()}
    assert set(rows) == {"meow", "purr", "woof"}
    assert (rows["meow"]["class"], rows["meow"]["_ludwig_split_idx"]) == ("cat", 0)
    assert (rows["purr"]["_ludwig_split_idx"], rows["woof"]["_ludwig_split_idx"]) == (1, 2)

    ctx = await snap.read_snapshot_manifest(version_id)
    assert ctx.class_counts == {"cat": 2, "dog": 1}
    assert sorted(ctx.label_class_names) == ["cat", "dog"]
    assert [c.name for c in ctx.columns][-1] == "_theseus_item_id"
    assert json.loads(fake_s3[f"snapshots/{version_id}/manifest.json"])["itemCount"] == 3


async def test_build_snapshot_handles_tabular_regression(db, fake_s3):
    async with db() as s:
        project, version = await _seed(s, "tabular_regression", "tabular")
        for age, target in [(30, 1.5), (40, 2.5)]:
            item = DatasetItem(dataset_id=project.id, content_hash=uuid.uuid4().hex + uuid.uuid4().hex)
            s.add(item)
            await s.flush()
            s.add_all(
                [
                    TabularFeatures(item_id=item.id, features_json={"age": age}),
                    DatasetVersionItem(version_id=version.id, item_id=item.id, split_type="train"),
                    Annotation(item_id=item.id, annotation_type="classification", label_structured={"value": target}),
                ]
            )
        await s.commit()
        version_id = version.id

    await snap.build_snapshot(version_id)
    table = pq.read_table(io.BytesIO(fake_s3[f"snapshots/{version_id}/dataset.parquet"]))
    assert sorted(r["target"] for r in table.to_pylist()) == [1.5, 2.5]
    ctx = await snap.read_snapshot_manifest(version_id)
    assert ctx.class_counts is None
    assert [c.name for c in ctx.columns if c.kind == "scalar"] == ["age"]


async def test_build_snapshot_marks_failed_on_error_and_only_from_building(db, fake_s3, monkeypatch):
    async with db() as s:
        _, version = await _seed(s, "text_classification", "text")
        await s.commit()
        version_id = version.id

    def boom(*a, **k):
        raise RuntimeError("s3 is down")

    monkeypatch.setattr(storage, "upload_bytes", boom)
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
        assert v.status == "failed" and "s3 is down" in v.failed_message

    # A later stray call must not resurrect a version that already left `building`.
    monkeypatch.setattr(storage, "upload_bytes", lambda *a, **k: None)
    await snap.build_snapshot(version_id)
    async with db() as s:
        assert (await s.get(DatasetVersion, version_id)).status == "failed"


async def test_build_snapshot_for_unknown_version_does_not_raise(db, fake_s3):
    await snap.build_snapshot(uuid.uuid4())
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetVersion))).scalar_one() == 0

"""Contract tests over every registered trainer backend, plus the generic parts of the plugin
base class itself (a backend that doesn't override anything beyond the abstract methods)."""

from theseus.backends.base import HyperparamsBase, ModelChoice, TrainerBackend
from theseus.backends.base import _registry as backend_registry
from theseus.backends.registry import list_backends
from theseus.export.registry import list_export_formats
from theseus.services.task_registry import TASK_REGISTRY, get_task_descriptor


def test_every_registered_backend_has_a_hyperparameters_model_and_sane_artifacts():
    for backend in list_backends():
        assert issubclass(backend.Hyperparameters, HyperparamsBase)
        for artifact_id, artifact in backend.artifacts.items():
            assert artifact.id == artifact_id and artifact.filename


def test_every_export_format_points_at_an_artifact_some_backend_can_produce():
    all_artifact_ids = {artifact_id for backend in list_backends() for artifact_id in backend.artifacts}
    for fmt in list_export_formats():
        assert fmt.artifact in all_artifact_ids, fmt.id


def test_every_task_is_supported_by_at_most_the_backends_that_claim_to():
    for task in TASK_REGISTRY.values():
        for backend in list_backends():
            if backend.supports(task):
                assert backend.hyperparameter_specs(task) is not None  # must not raise
                assert backend.models(task) is not None  # must not raise


def test_a_backend_that_only_implements_the_required_methods_still_works():
    # Defined inside the test (not at module level) so it registers itself, via
    # __init_subclass__, no earlier than this test's own setup — a module-level plugin class in a
    # test file registers at IMPORT time, which happens for every test file during collection,
    # before any test runs, and would leak into other files' `list_backends()` until this file's
    # tests finish executing.
    class _MinimalBackend(TrainerBackend):
        """Declares nothing beyond the abstract methods: exercises TrainerBackend's own default
        behavior (hyperparameter_specs, evaluate, preprocessing_manifest), not any real framework's."""

        id = "test_only_minimal_backend"
        label = "Minimal"

        @classmethod
        def supports(cls, task):
            return True

        @classmethod
        def models(cls, task):
            return [ModelChoice(id="only", label="Only")]

        @classmethod
        def compile(cls, task, ctx, hp):
            return {}

        @classmethod
        def load(cls, model_dir):
            raise NotImplementedError

        @classmethod
        def train(cls, run):
            raise NotImplementedError

        @classmethod
        def convert(cls, model, artifact_id, workdir):
            raise NotImplementedError

    try:
        task = get_task_descriptor("image_classification")

        # modelId is always excluded (models() covers it); batchSize mixes int and "auto" and has
        # no generic representation, so the lenient default skips it rather than raising.
        names = {p.name for p in _MinimalBackend.hyperparameter_specs(task)}
        assert names == {"epochs", "learningRate", "earlyStopPatience"}

        assert _MinimalBackend.evaluate(None, None, "split", "id") is None  # no evaluation support by default

        try:
            _MinimalBackend.preprocessing_manifest("r1", {}, None)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("expected NotImplementedError")
    finally:
        backend_registry.unregister("test_only_minimal_backend")


def test_param_specs_read_the_section_from_json_schema_extra():
    from pydantic import BaseModel, Field

    from theseus.params import param_specs

    class Params(BaseModel):
        lr: float = Field(0.1, ge=0, le=1, json_schema_extra={"group": "Optimisation", "step": 0.05})
        plain: int = 3

    specs = {p.name: p for p in param_specs(Params)}
    assert specs["lr"].group == "Optimisation" and specs["lr"].step == 0.05
    assert specs["plain"].group is None  # ungrouped parameters stay ungrouped

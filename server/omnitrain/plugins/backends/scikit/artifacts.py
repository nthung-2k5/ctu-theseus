"""Model artifacts: the conversions `SklearnBackend.convert` builds export bundles from. Only ONNX is offered.

The graph is shaped to what the generated clients already expect of a tabular model, so a scikit-learn run needs no
client of its own:

  * A tabular model takes one float input per column, named by the column, of shape `[batch]`, as raw values:
    imputation and scaling are part of the graph, so the manifest carries no `numberNormalization`.
  * A classifier outputs LOGITS, one tensor named by the label column. scikit-learn gives probabilities, so the graph
    ends in `log(max(p, 1e-9))`; the client's softmax turns that back into the very same probabilities.
  * A regressor outputs its number in one tensor named by the target column.
  * A text model takes one raw string tensor `[batch, 1]` and tokenises and weights it inside the graph, using the
    `com.microsoft.Tokenizer` operator, which only onnxruntime provides.

This module imports neither scikit-learn nor onnx at module level.
"""

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from omnitrain.plugins.backends.base import ARTIFACT_FILENAMES, Artifact
from omnitrain.plugins.backends.scikit import estimators

ARTIFACTS: dict[str, Artifact] = {"onnx": Artifact("onnx", ARTIFACT_FILENAMES["onnx"])}

OPSET = 17
# Probabilities are clipped to this before the log, so a probability of exactly 0 is a very negative logit, not -inf.
MIN_PROBABILITY = 1e-9
# The name of the single matrix input of the converted pipeline, before tabular columns are wired into it.
_FEATURES = "_features"


@contextlib.contextmanager
def _bool_attributes_as_ints() -> Iterator[None]:
    """skl2onnx hands `onnx.helper.make_attribute` a list that mixes bools and ints for a tree ensemble that handles
    missing values (`nodes_missing_value_tracks_true`), which current protobuf rejects. Turn those into ints for the
    duration of one conversion. Conversions are serialised by `jobs/export.py`, so patching a module global is safe."""
    import numpy as np
    from onnx import helper

    original = helper.make_attribute

    def patched(key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, (list, tuple, np.ndarray)) and any(isinstance(v, (bool, np.bool_)) for v in value):
            value = [int(v) for v in value]
        return original(key, value, *args, **kwargs)

    helper.make_attribute = patched
    try:
        yield
    finally:
        helper.make_attribute = original


def convert(model: Any, artifact_id: str, workdir: str) -> str:
    """Convert `model` (a `SklearnLoadedModel`) and return the path of the written file."""
    import onnx

    artifact = ARTIFACTS[artifact_id]
    onnx_model = to_onnx(model.estimator, model.meta)
    onnx.checker.check_model(onnx_model)

    out_dir = os.path.join(workdir, artifact.id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, artifact.filename)
    onnx.save(onnx_model, path)
    return path


def to_onnx(estimator: Any, meta: dict[str, Any]) -> Any:
    """The ONNX model for a fitted pipeline, shaped as the module docstring describes."""
    import numpy as np
    from skl2onnx import to_onnx as convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    classify = meta["task"] == "classify"
    columns: list[str] = meta["inputColumns"]
    text = meta["inputKind"] == "text"
    if text and getattr(estimator.named_steps["tfidf"], "analyzer", "word") != "word":
        raise RuntimeError(
            "Only a text model that splits into words can be exported to ONNX: scikit-learn's character n-grams have "
            "no ONNX equivalent. Train it again with the word analyzer."
        )

    if text:
        initial = [(columns[0], StringTensorType([None, 1]))]
        sample = np.array([["text"]])
    else:
        initial = [(_FEATURES, FloatTensorType([None, len(columns)]))]
        sample = np.zeros((1, len(columns)), dtype=np.float32)
    final = estimators.final_estimator(estimator)
    options = {id(final): {"zipmap": False}} if classify else None
    with _bool_attributes_as_ints():
        graph_model = convert_sklearn(
            estimator, sample, initial_types=initial, options=options, target_opset={"": OPSET, "ai.onnx.ml": 3}
        )

    _shape_outputs(graph_model, meta["outputName"], classify)
    if not text:
        _split_inputs(graph_model, columns)
    return graph_model


def _shape_outputs(model: Any, name: str, classify: bool) -> None:
    """Leave exactly one output, named `name`: the logits of a classifier, the number of a regressor."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    produced = {t for node in graph.node for t in node.output} | {i.name for i in graph.input}
    if name in produced:
        raise RuntimeError(f'The label column "{name}" clashes with an internal tensor of the converted model')

    wanted = "probabilities" if classify else "variable"
    source = next((o for o in graph.output if o.name == wanted), None)
    if source is None:
        raise RuntimeError(f"The converted model has no {wanted!r} output (it has {[o.name for o in graph.output]})")

    if classify:
        floor = f"{name}__floor"
        graph.initializer.append(numpy_helper.from_array(np.array(MIN_PROBABILITY, dtype=np.float32), floor))
        clipped = f"{name}__clipped"
        graph.node.append(helper.make_node("Max", [wanted, floor], [clipped], name=f"{name}__max"))
        graph.node.append(helper.make_node("Log", [clipped], [name], name=f"{name}__log"))
    else:
        graph.node.append(helper.make_node("Identity", [wanted], [name], name=f"{name}__out"))

    shape = [d.dim_value if d.dim_value else d.dim_param or None for d in source.type.tensor_type.shape.dim]
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))


def _split_inputs(model: Any, columns: list[str]) -> None:
    """Replace the one `[batch, n]` input with one `[batch]` input per column, concatenated back inside the graph."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    taken = {t for node in graph.node for t in node.output} | {i.name for i in graph.initializer}
    clashes = sorted(set(columns) & (taken | {_FEATURES}))
    if clashes:
        raise RuntimeError(f"These column names clash with internal tensors of the converted model: {clashes}")
    if len(set(columns)) != len(columns):
        raise RuntimeError("Two input columns have the same name")

    original = next(i for i in graph.input if i.name == _FEATURES)
    batch = original.type.tensor_type.shape.dim[0].dim_param or None

    axes = f"{_FEATURES}__axes"
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), axes))
    columns_2d: list[str] = []
    new_nodes = []
    for column in columns:
        column_2d = f"{column}__2d"
        new_nodes.append(helper.make_node("Unsqueeze", [column, axes], [column_2d], name=f"{column}__unsqueeze"))
        columns_2d.append(column_2d)
    new_nodes.append(helper.make_node("Concat", columns_2d, [_FEATURES], axis=1, name=f"{_FEATURES}__concat"))

    # The new nodes feed an existing tensor, so they go first to keep the node list topologically sorted.
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes + existing)
    del graph.input[:]
    graph.input.extend(helper.make_tensor_value_info(c, TensorProto.FLOAT, [batch]) for c in columns)

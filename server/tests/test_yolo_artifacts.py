import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from theseus.backends.base import ARTIFACT_FILENAMES
from theseus.backends.yolo import YoloBackend
from theseus.backends.yolo.artifacts import ARTIFACTS, strip_final_softmax

W = np.array([[1.0, -1.0], [0.5, 2.0], [-1.0, 0.5]], dtype=np.float32)
B = np.array([0.1, -0.2], dtype=np.float32)


def build_graph(path, final_op):
    """x -> Gemm -> (Softmax|Relu) -> output0: the tail of an Ultralytics classification export."""
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["logits"])]
    nodes.append(helper.make_node(final_op, ["logits"], ["output0"]))
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(W, "w"), numpy_helper.from_array(B, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def run(path, x):
    session = ort.InferenceSession(str(path))
    return session.run(None, {"x": x})[0]


X = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)


def test_strip_final_softmax_turns_probabilities_into_the_logits_behind_them(tmp_path):
    path = tmp_path / "m.onnx"
    build_graph(path, "Softmax")
    probabilities = run(path, X)
    assert probabilities.sum() == pytest.approx(1.0)

    strip_final_softmax(str(path))

    logits = run(path, X)
    np.testing.assert_allclose(logits, X @ W + B, rtol=1e-5)
    # The generated clients softmax the output themselves; that must give the original probabilities back.
    exp = np.exp(logits - logits.max())
    np.testing.assert_allclose(exp / exp.sum(), probabilities, rtol=1e-5)
    graph = onnx.load(str(path)).graph
    assert [o.name for o in graph.output] == ["output0"]
    assert [n.op_type for n in graph.node] == ["Gemm"]


def test_a_graph_that_does_not_end_in_softmax_is_refused_not_guessed_at(tmp_path):
    path = tmp_path / "m.onnx"
    build_graph(path, "Relu")
    with pytest.raises(RuntimeError, match="ends in Relu"):
        strip_final_softmax(str(path))


def test_the_backend_offers_onnx_only_under_the_shared_filename():
    assert list(ARTIFACTS) == ["onnx"]
    assert ARTIFACTS["onnx"].filename == ARTIFACT_FILENAMES["onnx"] == "model.onnx"
    assert YoloBackend.artifacts is ARTIFACTS

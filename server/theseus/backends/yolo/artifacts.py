"""Model artifacts: the conversions `YoloBackend.convert` builds export bundles from.

Only ONNX is offered. Several export formats share one artifact (every devkit and app ships the ONNX model),
so conversion is keyed by artifact, not by format: it runs once per run and is reused (see jobs/export.py).
This module imports neither ultralytics nor torch at module level.
"""

import os
import shutil
from typing import Any

from theseus.backends.base import ARTIFACT_FILENAMES, Artifact

ARTIFACTS: dict[str, Artifact] = {"onnx": Artifact("onnx", ARTIFACT_FILENAMES["onnx"])}


def convert(model: Any, artifact_id: str, workdir: str) -> str:
    """Export `model` (a `YoloLoadedModel`) and return the path of the written file.

    Ultralytics writes the exported file next to the weights it was loaded from. That is the downloaded model
    cache, which is shared with serving, so the export runs on a private copy under `workdir` instead.
    """
    from theseus.backends.yolo import runtime

    artifact = ARTIFACTS[artifact_id]
    runtime.configure()
    from ultralytics import YOLO

    scratch = os.path.join(workdir, "yolo-export")
    os.makedirs(scratch, exist_ok=True)
    weights = os.path.join(scratch, runtime.WEIGHTS_FILENAME)
    shutil.copyfile(model.yolo.ckpt_path, weights)

    # simplify=False: the simplifier is an optional package Ultralytics would try to pip install.
    # nms=False and dynamic=False: the plain, fixed-size graph every generated client can run.
    exported = YOLO(weights).export(
        format="onnx", imgsz=model.meta["imgsz"], simplify=False, dynamic=False, nms=False, device="cpu"
    )

    out_dir = os.path.join(workdir, artifact.id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, artifact.filename)
    shutil.move(exported, path)
    if not os.path.isfile(path):
        raise RuntimeError(f"Ultralytics did not write {artifact.filename} for artifact {artifact_id!r}")
    strip_final_softmax(path)
    return path


def strip_final_softmax(path: str) -> None:
    """Make the graph at `path` output logits instead of probabilities, in place.

    Ultralytics' classification head ends in a Softmax when exporting. Every generated client applies its own
    softmax to the model output (the Ludwig graph ends in `::logits`), so left alone the probabilities would be
    softmaxed twice: the ranking survives but every confidence is squashed towards uniform and no longer matches
    `expected.json`. Raises rather than guessing if the graph does not end in a Softmax.
    """
    import onnx

    model = onnx.load(path)
    graph = model.graph
    output = graph.output[0].name
    softmax = next((n for n in graph.node if output in n.output), None)
    if softmax is None or softmax.op_type != "Softmax":
        found = softmax.op_type if softmax is not None else "nothing"
        raise RuntimeError(f"Expected the exported YOLO graph to end in a Softmax, but it ends in {found}")

    logits = softmax.input[0]
    graph.node.remove(softmax)
    # Whatever produced the logits now produces the graph output itself.
    for node in graph.node:
        for i, name in enumerate(node.output):
            if name == logits:
                node.output[i] = output
        for i, name in enumerate(node.input):
            if name == logits:
                node.input[i] = output
    onnx.save(model, path)

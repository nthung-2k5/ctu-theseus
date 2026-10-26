from omnitrain.plugins.export.formats._kits import AUDIO, DETECTION, STANDARD, VIDEO, DevkitFormat
from omnitrain.plugins.export.readme import (
    AUDIO_LIMITATION,
    DETECTION_LIMITATION,
    HF_TEXT_LIMITATION,
    LIMITATION_FULL,
    TEXT_LIMITATION,
)


class PythonDevkit(DevkitFormat):
    id = "python_devkit"
    label = "Python devkit"
    description = "ONNX model plus a Python inference client (onnxruntime), example and verify script."
    order = 10
    handles = frozenset({STANDARD, DETECTION, VIDEO, AUDIO})
    readme_info = {
        "label": "Python",
        "files": "`omnitrain_client.py`, `example.py`",
        "dependencies": "pip install onnxruntime numpy pillow",
        "run_cmd": "python example.py <path-to-image>\npython example.py '{\"column\": value, ...}'   # tabular models",
        "verify_cmd": "python verify.py",
        "limitation": LIMITATION_FULL,
        "kinds": {
            "detection": {
                "run_cmd": (
                    "python example.py <path-to-image>   # one line per box: label, confidence, x y w h"
                    "\n# in code: client.detect(path, confidence=, iou=, max_detections=)"
                ),
                "limitation": DETECTION_LIMITATION,
            },
            "audio": {
                "dependencies": (
                    "pip install onnxruntime numpy pillow\npip install soundfile   # optional: formats other than WAV"
                ),
                "run_cmd": "python example.py <path-to-audio-file>",
                "limitation": AUDIO_LIMITATION,
            },
            "text": {
                "dependencies": "pip install onnxruntime numpy",
                "run_cmd": 'python example.py "<the text to classify>"',
                "limitation": TEXT_LIMITATION,
            },
            "hf_text": {
                "dependencies": "pip install onnxruntime numpy tokenizers",
                "run_cmd": 'python example.py "<the text to classify>"',
                "files": "`omnitrain_client.py`, `example.py`, `tokenizer.json`",
                "limitation": HF_TEXT_LIMITATION,
            },
            "video": {
                "dependencies": "pip install onnxruntime numpy pillow opencv-python-headless",
                "run_cmd": "python example.py <path-to-video>",
                "limitation": (
                    "Video classification: `predict(path)` decodes the file with OpenCV, takes "
                    "`inputs[0].videoSampling.clipsPerVideo` clips of `numFrames` frames spread over the video, "
                    "scores each, and averages the clips' probabilities. Decoding and frame resizing use OpenCV, "
                    "the same library the platform used to train; another decoder can pick slightly different "
                    "frames and change the scores."
                ),
            },
        },
    }
    client_files = (
        ("omnitrain_client.py", "python/omnitrain_client.py"),
        ("example.py", "python/example.py"),
    )
    verify_files = (("verify.py", "python/verify.py"),)

from theseus.export.formats._kits import DevkitFormat
from theseus.export.readme import LIMITATION_FULL


class PythonDevkit(DevkitFormat):
    id = "python_devkit"
    label = "Python devkit"
    description = "ONNX model plus a Python inference client (onnxruntime), example and verify script."
    order = 10
    readme_info = {
        "label": "Python",
        "files": "`theseus_client.py`, `example.py`",
        "dependencies": "pip install onnxruntime numpy pillow",
        "run_cmd": "python example.py <path-to-image>\npython example.py '{\"column\": value, ...}'   # tabular models",
        "verify_cmd": "python verify.py",
        "limitation": LIMITATION_FULL,
    }
    client_files = (
        ("theseus_client.py", "python/theseus_client.py"),
        ("example.py", "python/example.py"),
    )
    verify_files = (("verify.py", "python/verify.py"),)

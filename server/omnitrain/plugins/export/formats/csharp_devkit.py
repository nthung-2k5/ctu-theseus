from omnitrain.plugins.export.formats._kits import AUDIO, DETECTION, STANDARD, DevkitFormat
from omnitrain.plugins.export.readme import (
    AUDIO_LIMITATION,
    DETECTION_LIMITATION,
    HF_TEXT_OTHER_LIMITATION,
    LIMITATION_IMAGE_ONLY,
    TEXT_LIMITATION,
)


class CSharpDevkit(DevkitFormat):
    id = "csharp_devkit"
    label = "C# devkit"
    description = "ONNX model plus a C# inference client (Microsoft.ML.OnnxRuntime) and example."
    order = 30
    handles = frozenset({STANDARD, DETECTION, AUDIO})
    readme_info = {
        "label": "C#",
        "files": "`OmniTrainClient.cs`, `Program.cs`",
        "dependencies": (
            "dotnet add package Microsoft.ML.OnnxRuntime --version 1.19.2\n"
            "dotnet add package SixLabors.ImageSharp --version 3.2.0"
        ),
        "run_cmd": "dotnet run -- <path-to-image>",
        "verify_cmd": "dotnet run -- verify",
        "limitation": LIMITATION_IMAGE_ONLY,
        "kinds": {
            "detection": {"limitation": DETECTION_LIMITATION},
            "audio": {"run_cmd": "dotnet run -- <path-to-wav-file>", "limitation": AUDIO_LIMITATION},
            "hf_text": {"limitation": HF_TEXT_OTHER_LIMITATION},
            "text": {"run_cmd": 'dotnet run -- "<the text to classify>"', "limitation": TEXT_LIMITATION},
        },
        "note": (
            "`Program.cs` uses top-level statements as its entry point. If your project already has one, "
            "merge its logic in rather than copying the file verbatim."
        ),
    }
    client_files = (
        ("OmniTrainClient.cs", "csharp/OmniTrainClient.cs"),
        ("Program.cs", "csharp/Program.cs"),
    )

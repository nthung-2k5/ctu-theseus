from theseus.export.formats._kits import DevkitFormat
from theseus.export.readme import LIMITATION_IMAGE_ONLY


class CSharpDevkit(DevkitFormat):
    id = "csharp_devkit"
    label = "C# devkit"
    description = "ONNX model plus a C# inference client (Microsoft.ML.OnnxRuntime) and example."
    order = 30
    readme_info = {
        "label": "C#",
        "files": "`TheseusClient.cs`, `Program.cs`",
        "dependencies": (
            "dotnet add package Microsoft.ML.OnnxRuntime --version 1.19.2\n"
            "dotnet add package SixLabors.ImageSharp --version 3.1.12"
        ),
        "run_cmd": "dotnet run -- <path-to-image>",
        "verify_cmd": "dotnet run -- verify",
        "limitation": LIMITATION_IMAGE_ONLY,
        "note": (
            "`Program.cs` uses top-level statements as its entry point. If your project already has one, "
            "merge its logic in rather than copying the file verbatim."
        ),
    }
    client_files = (
        ("TheseusClient.cs", "csharp/TheseusClient.cs"),
        ("Program.cs", "csharp/Program.cs"),
    )

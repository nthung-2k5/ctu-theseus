from omnitrain.plugins.export.formats._kits import AUDIO, DETECTION, STANDARD, DevkitFormat
from omnitrain.plugins.export.readme import AUDIO_LIMITATION, DETECTION_LIMITATION, LIMITATION_FULL


class TypeScriptDevkit(DevkitFormat):
    id = "typescript_devkit"
    label = "TypeScript devkit (Bun)"
    description = "ONNX model plus a TypeScript inference client (onnxruntime-node), example and verify script."
    order = 20
    handles = frozenset({STANDARD, DETECTION, AUDIO})
    readme_info = {
        "label": "TypeScript (Bun)",
        "files": "`client.ts`, `example.ts`",
        "dependencies": "bun add onnxruntime-node sharp",
        "run_cmd": "bun example.ts <path-to-image>\nbun example.ts '{\"column\": value, ...}'   # tabular models",
        "verify_cmd": "bun verify.ts",
        "limitation": LIMITATION_FULL,
        "kinds": {
            "audio": {"run_cmd": "bun example.ts <path-to-wav-file>", "limitation": AUDIO_LIMITATION},
            "detection": {
                "run_cmd": "bun example.ts <path-to-image>   # one line per box: label, confidence, x y w h",
                "limitation": DETECTION_LIMITATION,
            },
        },
    }
    client_files = (
        ("client.ts", "typescript/client.ts.tmpl"),
        ("example.ts", "typescript/example.ts.tmpl"),
    )
    verify_files = (("verify.ts", "typescript/verify.ts.tmpl"),)

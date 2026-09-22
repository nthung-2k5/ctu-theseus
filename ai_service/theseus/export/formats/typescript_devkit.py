from theseus.export.formats._kits import DevkitFormat
from theseus.export.readme import LIMITATION_FULL


class TypeScriptDevkit(DevkitFormat):
    id = "typescript_devkit"
    label = "TypeScript devkit (Bun)"
    description = "ONNX model plus a TypeScript inference client (onnxruntime-node), example and verify script."
    order = 20
    readme_info = {
        "label": "TypeScript (Bun)",
        "files": "`client.ts`, `example.ts`",
        "dependencies": "bun add onnxruntime-node sharp",
        "run_cmd": "bun example.ts <path-to-image>\nbun example.ts '{\"column\": value, ...}'   # tabular models",
        "verify_cmd": "bun verify.ts",
        "limitation": LIMITATION_FULL,
    }
    client_files = (
        ("client.ts", "typescript/client.ts.tmpl"),
        ("example.ts", "typescript/example.ts.tmpl"),
    )
    verify_files = (("verify.ts", "typescript/verify.ts.tmpl"),)

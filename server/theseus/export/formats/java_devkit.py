from theseus.export.formats._kits import DevkitFormat
from theseus.export.readme import LIMITATION_IMAGE_ONLY


class JavaDevkit(DevkitFormat):
    id = "java_devkit"
    label = "Java / Kotlin devkit"
    description = "ONNX model plus a Java inference client (onnxruntime) that Kotlin can call directly."
    order = 40
    readme_info = {
        "label": "Java / Kotlin",
        "files": "`TheseusClient.java`, `Main.java`",
        "dependencies": (
            "// Gradle (Kotlin DSL)\n"
            'implementation("com.microsoft.onnxruntime:onnxruntime:1.29.0")\n'
            'implementation("com.fasterxml.jackson.core:jackson-databind:2.22.1")'
        ),
        "run_cmd": "java Main <path-to-image>",
        "verify_cmd": "java Main verify",
        "limitation": LIMITATION_IMAGE_ONLY,
        "note": "Written in Java for maximum interop: call it directly from Kotlin with zero wrapping.",
    }
    client_files = (
        ("TheseusClient.java", "java/TheseusClient.java"),
        ("Main.java", "java/Main.java"),
    )

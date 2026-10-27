from omnitrain.plugins.export.formats._kits import AUDIO, DETECTION, STANDARD, DevkitFormat
from omnitrain.plugins.export.readme import (
    AUDIO_LIMITATION,
    DETECTION_LIMITATION,
    HF_TAGS_OTHER_LIMITATION,
    HF_TEXT_OTHER_LIMITATION,
    LIMITATION_IMAGE_ONLY,
    TEXT_LIMITATION,
    tags_limitation,
)


class JavaDevkit(DevkitFormat):
    id = "java_devkit"
    label = "Java / Kotlin devkit"
    description = "ONNX model plus a Java inference client (onnxruntime) that Kotlin can call directly."
    order = 40
    handles = frozenset({STANDARD, DETECTION, AUDIO})
    readme_info = {
        "label": "Java / Kotlin",
        "files": "`OmniTrainClient.java`, `Main.java`",
        "dependencies": (
            "// Gradle (Kotlin DSL)\n"
            'implementation("com.microsoft.onnxruntime:onnxruntime:1.29.0")\n'
            'implementation("com.fasterxml.jackson.core:jackson-databind:2.22.1")'
        ),
        "run_cmd": "java Main <path-to-image>",
        "verify_cmd": "java Main verify",
        "limitation": LIMITATION_IMAGE_ONLY,
        "kinds": {
            "detection": {"limitation": DETECTION_LIMITATION},
            "audio": {"run_cmd": "java Main <path-to-wav-file>", "limitation": AUDIO_LIMITATION},
            "hf_text": {"limitation": HF_TEXT_OTHER_LIMITATION},
            "tags": {
                "run_cmd": 'java Main "<the text to tag>"   # one line per word: word, tag, confidence',
                "limitation": tags_limitation("tag"),
            },
            "hf_tags": {"limitation": HF_TAGS_OTHER_LIMITATION},
            "text": {"run_cmd": 'java Main "<the text to classify>"', "limitation": TEXT_LIMITATION},
        },
        "note": "Written in Java for maximum interop: call it directly from Kotlin with zero wrapping.",
    }
    client_files = (
        ("OmniTrainClient.java", "java/OmniTrainClient.java"),
        ("Main.java", "java/Main.java"),
    )

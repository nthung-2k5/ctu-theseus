from omnitrain.plugins.export.common import render, template
from omnitrain.plugins.export.formats._kits import AUDIO, DETECTION, STANDARD, AppFormat
from omnitrain.plugins.export.formats.base import BundleContext
from omnitrain.plugins.export.readme import AUDIO_APP_LIMITATION, DETECTION_LIMITATION


class PwaApp(AppFormat):
    id = "pwa_app"
    label = "Progressive Web App"
    description = "An installable in-browser app running the ONNX model on-device. No server."
    order = 10
    handles = frozenset({STANDARD, DETECTION, AUDIO})
    readme_info = {
        "label": "Progressive Web App",
        "setup_cmd": "python -m http.server 8000   # or: npx serve .",
        "run_cmd": 'Open http://localhost:8000 in a browser, then use the browser\'s "Install" prompt.',
        "verify_note": 'Use the "Run built-in self-check" button on the page.',
        "kinds": {"detection": {"limitation": DETECTION_LIMITATION}, "audio": {"limitation": AUDIO_APP_LIMITATION}},
    }

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        variables = {"RUN_NAME": ctx.run_name, "TASK": ctx.task_label}
        # Fetched relative to index.html at runtime, so root placement is correct.
        ctx.place_model()
        ctx.place_golden()
        ctx.add("index.html", render(template("pwa/index.html.tmpl"), variables))
        ctx.add("app.js", template("pwa/app.js"))
        ctx.add("sw.js", template("pwa/sw.js"))
        ctx.add("manifest.webmanifest", render(template("pwa/manifest.webmanifest.tmpl"), variables))
        ctx.add("icon.svg", template("pwa/icon.svg"))
        ctx.add("style.css", template("pwa/style.css"))
        ctx.add("README.md", cls.readme(ctx))

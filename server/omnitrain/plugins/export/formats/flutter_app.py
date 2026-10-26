from omnitrain.plugins.export.common import dart_package_name, render, template
from omnitrain.plugins.export.formats._kits import DETECTION, STANDARD, AppFormat
from omnitrain.plugins.export.formats.base import BundleContext
from omnitrain.plugins.export.readme import DETECTION_LIMITATION


class FlutterApp(AppFormat):
    id = "flutter_app"
    label = "Flutter app"
    description = "A Flutter project running the ONNX model on-device (Android, iOS, desktop). No server."
    order = 20
    handles = frozenset({STANDARD, DETECTION})
    readme_info = {
        "label": "Flutter app",
        "setup_cmd": "flutter pub get",
        "run_cmd": "flutter run   # or: flutter build apk / flutter build ios",
        "verify_note": "Tap the checkmark icon in the app bar to run the built-in self-check screen.",
        "kinds": {"detection": {"limitation": DETECTION_LIMITATION}},
    }

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        variables = {"RUN_NAME": ctx.run_name, "TASK": ctx.task_label}
        # Flutter only reads files declared as pubspec assets, so the assets/ prefix is required.
        ctx.place_model("assets/")
        ctx.place_golden("assets/")
        ctx.add(
            "pubspec.yaml",
            render(
                template("flutter/pubspec.yaml.tmpl"),
                {**variables, "PACKAGE_NAME": dart_package_name(ctx.run_name)},
            ),
        )
        ctx.add("lib/main.dart", render(template("flutter/lib/main.dart.tmpl"), variables))
        ctx.add("lib/omnitrain_client.dart", template("flutter/lib/omnitrain_client.dart"))
        ctx.add("README.md", cls.readme(ctx))

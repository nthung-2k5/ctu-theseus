from theseus.export.common import dart_package_name, render, template
from theseus.export.formats._kits import AppFormat
from theseus.export.formats.base import BundleContext


class FlutterApp(AppFormat):
    id = "flutter_app"
    label = "Flutter app"
    description = "A Flutter project running the ONNX model on-device (Android, iOS, desktop). No server."
    order = 20
    readme_info = {
        "label": "Flutter app",
        "setup_cmd": "flutter pub get",
        "run_cmd": "flutter run   # or: flutter build apk / flutter build ios",
        "verify_note": "Tap the checkmark icon in the app bar to run the built-in self-check screen.",
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
        ctx.add("lib/theseus_client.dart", template("flutter/lib/theseus_client.dart"))
        ctx.add("README.md", cls.readme(ctx))

from theseus.export.formats.base import BundleContext, ExportFormat


class OnnxModel(ExportFormat):
    id = "onnx"
    label = "ONNX model"
    description = "The bare .onnx model with preprocessing.json and labels.txt. No client code."
    group = "Model"
    order = 10
    artifact = "onnx"

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        ctx.place_model()
        ctx.add("README.md", cls.readme(ctx))

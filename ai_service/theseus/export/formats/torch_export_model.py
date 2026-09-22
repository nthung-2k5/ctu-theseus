from theseus.export.formats.base import BundleContext, ExportFormat


class TorchExportModel(ExportFormat):
    id = "torch_export"
    label = "PyTorch (torch.export)"
    description = "A torch.export program (.pt2) with preprocessing.json and labels.txt. Needs PyTorch to run."
    group = "Model"
    order = 20
    artifact = "torch_export"

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        ctx.place_model()
        ctx.add("README.md", cls.readme(ctx))

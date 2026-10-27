import io
import zipfile

from omnitrain.domain.task_registry import TaskDescriptor
from omnitrain.plugins.export.common import dumps, template
from omnitrain.plugins.export.formats.base import BundleContext, ExportFormat

README = """# {run_name}

Exported from OmniTrain: task **{task_label}**, format **{format_label}**.

A language model fine-tuned on your data, as a Hugging Face model folder with a Python client around it. It writes text
for text: nothing here is an ONNX graph, because a generative model has no single fixed-shape graph to export.

## Contents

- `model/`: the whole model (`config.json`, `model.safetensors`, the tokenizer files, `names.json`). The LoRA adapters
  were merged into the weights, so it loads with `transformers` alone, no `peft` and no base model. Open it with
  `AutoModelForCausalLM.from_pretrained("model")` in any tool that reads Hugging Face models.
- `preprocessing.json`: how a prompt is built from the input fields and how the answer is decoded (`generation`).
- `omnitrain_llm.py`, `example.py`: a client that repeats that recipe.{verify_files}

## Limitation

The platform asks the model for its most likely continuation (greedy decoding, no sampling) and stops at the
end-of-text token or after `generation.maxNewTokens` new tokens. A prompt longer than `generation.promptRoom` tokens
loses its start. Only a client that does the same writes the same answers; the layout and the cut are spelled out in
`preprocessing.json`. The platform runs a model in bfloat16 on a GPU that has it natively and in float32 otherwise;
`omnitrain_llm.py` does the same, and another number format may rarely change an answer.

## Dependencies

```
pip install torch transformers
```

## Quick start

```bash
python example.py "<the input text>"
python example.py '{{"field": "text", ...}}'   # a model with several input fields
```{verify}
"""

VERIFY = """

## Verify

This bundle includes one real test-split sample and the platform's own answer for it (`expected.json`). Run:

```bash
python verify.py
```

to confirm this client writes the same text."""


class HfModel(ExportFormat):
    id = "hf_model"
    label = "Hugging Face model"
    description = (
        "The fine-tuned language model as a Hugging Face folder, with a Python client, example and verify script."
    )
    notice = (
        "Python only. The model is loaded with torch and transformers, so it is large (hundreds of megabytes or more)."
    )
    group = "Model"
    order = 30
    artifact = "hf_model"
    needs_golden = True

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.output is not None and task.output.kind == "text"

    @classmethod
    def assemble(cls, ctx: BundleContext) -> None:
        with zipfile.ZipFile(io.BytesIO(ctx.model_bytes)) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                # Weights barely compress; the small JSON files do.
                ctx.add(f"model/{name}", archive.read(name), compress=not name.endswith(".safetensors"))
        ctx.add("preprocessing.json", dumps(ctx.preprocessing))
        ctx.place_golden()
        ctx.add("omnitrain_llm.py", template("python_llm/omnitrain_llm.py"))
        ctx.add("example.py", template("python_llm/example.py"))
        if ctx.golden is not None:
            ctx.add("verify.py", template("python_llm/verify.py"))
        ctx.add("README.md", cls.readme(ctx))

    @classmethod
    def readme(cls, ctx: BundleContext) -> str:
        has_verify = ctx.golden is not None
        return README.format(
            run_name=ctx.run_name,
            task_label=ctx.task_label,
            format_label=ctx.format_label,
            verify_files=", `verify.py`, `expected.json`, `sample/`" if has_verify else "",
            verify=VERIFY if has_verify else "",
        )

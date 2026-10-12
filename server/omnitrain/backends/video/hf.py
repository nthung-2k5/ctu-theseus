"""Hugging Face video classifiers as a network the rest of the backend can train, serve and export.

Imports torch (and transformers inside functions): only loaded where a model is really built. Nothing here ever
trusts remote code or unpickles weights: local files only, safetensors only.

The backend's tensors are channels-first (N x 3 x T x H x W, what torchvision video models and the exported ONNX graph
take); transformers wants N x T x 3 x H x W. `HFVideoNet` is that one permute and a `.logits`, so everything else
(the training loop, ONNX export, serving) treats a Hugging Face model like any other network.
"""

from pathlib import Path

import torch
from torch import nn

# Parameter-name fragments of the classification head (and the norm in front of it). Freezing the backbone keeps
# everything else fixed.
_HEAD = ("classifier", "fc_norm")


class HFVideoNet(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=clips.permute(0, 2, 1, 3, 4)).logits


def _auto_class():
    from transformers import AutoModelForVideoClassification

    return AutoModelForVideoClassification


def build(path: str, names: list[str]) -> HFVideoNet:
    """The checkpoint at `path` with a fresh `len(names)`-way classification head.

    The backbone is loaded strictly, with the checkpoint's own config: `ignore_mismatched_sizes` would not only allow
    a different head, it would quietly re-initialise ANY tensor whose shape differs from the config, so a corrupt or
    mislabelled checkpoint would train from random weights while claiming to be pretrained. The head is replaced
    afterwards, here, because it is the one part that is meant not to match.
    """
    model = _auto_class().from_pretrained(path, local_files_only=True, use_safetensors=True, trust_remote_code=False)
    config = model.config
    head = nn.Linear(config.hidden_size, len(names))
    nn.init.normal_(head.weight, std=getattr(config, "initializer_range", 0.02))
    nn.init.zeros_(head.bias)
    model.classifier = head
    model.num_labels = config.num_labels = len(names)
    config.id2label = dict(enumerate(names))
    config.label2id = {name: i for i, name in enumerate(names)}
    return HFVideoNet(model)


def freeze_backbone(net: HFVideoNet) -> None:
    for name, param in net.named_parameters():
        param.requires_grad = any(part in name for part in _HEAD)


def save(net: HFVideoNet, dest: str) -> None:
    """A self-contained folder (config.json plus model.safetensors) `load` can read with no other file."""
    net.model.save_pretrained(dest, safe_serialization=True)


def load(path: str) -> HFVideoNet:
    model = _auto_class().from_pretrained(path, local_files_only=True, use_safetensors=True, trust_remote_code=False)
    return HFVideoNet(model)


def smoke_test(path: str) -> None:
    """Build the model with a two-class head and push one blank clip through it. Raises if anything is wrong."""
    from omnitrain.backends.video.custom import clip_settings

    settings = clip_settings(Path(path))
    net = build(path, ["a", "b"]).eval()
    clip = torch.zeros(1, 3, settings["numFrames"], settings["frameSize"], settings["frameSize"])
    with torch.no_grad():
        logits = net(clip)
    if tuple(logits.shape) != (1, 2):
        raise ValueError(f"expected one score per class, got shape {tuple(logits.shape)}")

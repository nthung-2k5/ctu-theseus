"""Build the networks: a torchvision image classifier with a fresh head, a multilayer perceptron for a table, and a
spectrogram classifier for a sound.

Imports torch (lazily imported module)."""

import torch
from torch import nn
from torchvision import models as tv

from omnitrain.plugins.backends.pytorch import tasks
from omnitrain.plugins.backends.pytorch.spec import TabularSpec


def _resolve(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    """(the module owning the layer at dotted `path`, the layer's name in it); `classifier.1` -> (classifier, "1")."""
    parent = root
    *parents, name = path.split(".")
    for part in parents:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, name


def _get(parent: nn.Module, name: str) -> nn.Module:
    return parent[int(name)] if name.isdigit() else getattr(parent, name)  # type: ignore[index]


def _set(parent: nn.Module, name: str, layer: nn.Module) -> None:
    if name.isdigit():
        parent[int(name)] = layer  # type: ignore[index]
    else:
        setattr(parent, name, layer)


def build_model(model_id: str, num_classes: int, *, pretrained: bool, freeze_backbone: bool = False) -> nn.Module:
    model = tasks.MODELS_BY_ID.get(model_id)
    if model is None:
        raise ValueError(f'"{model_id}" is not an image model')
    weights = getattr(tv, model.weights).DEFAULT if pretrained else None
    net = getattr(tv, model.builder)(weights=weights)
    parent, name = _resolve(net, model.head)
    old = _get(parent, name)
    if not isinstance(old, nn.Linear):
        raise ValueError(f"{model.label}'s head at {model.head} is a {type(old).__name__}, not a Linear layer")
    head = nn.Linear(old.in_features, num_classes)
    _set(parent, name, head)
    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return net


class TabularNet(nn.Module):
    """A multilayer perceptron over a row of numbers that does its own preprocessing, so the exported graph takes raw
    values: a cell that is not finite is replaced by the column's `fill`, then `(x - mean) / std` (all from the
    `TabularSpec`, fitted on the train split). A regressor outputs the target in its own units; the network itself
    works in standardised ones and the graph un-does that."""

    def __init__(self, spec: TabularSpec, hidden: list[int], dropout: float, outputs: int, *, regress: bool) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(spec.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(spec.std, dtype=torch.float32))
        self.register_buffer("fill", torch.tensor(spec.fill, dtype=torch.float32))
        self.regress = regress
        self.target_mean = spec.target_mean
        self.target_std = spec.target_std
        layers: list[nn.Module] = []
        width = len(spec.columns)
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1 if regress else outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, self.fill)
        out = self.layers((x - self.mean) / self.std)
        if self.regress:
            return out.squeeze(-1) * self.target_std + self.target_mean
        return out


class NamedColumns(nn.Module):
    """Wraps a `TabularNet` so it takes one tensor per column, named by the column, the way the generated clients feed
    it (see manifest.py). Only for export: training and serving use the `[rows, columns]` matrix."""

    def __init__(self, net: TabularNet, columns: tuple[str, ...]) -> None:
        super().__init__()
        self.net = net
        self.columns = columns

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(torch.stack([inputs[c] for c in self.columns], dim=1))


def build_tabular(spec: TabularSpec, hidden: list[int], dropout: float, outputs: int, *, regress: bool) -> TabularNet:
    return TabularNet(spec, hidden, dropout, outputs, regress=regress)


class AudioCnn(nn.Module):
    """Four convolution blocks over a one-channel `[mel bands, frames]` spectrogram, trained from scratch. Global
    average pooling makes it indifferent to the exact spectrogram size."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = 1
        for channels in (16, 32, 64, 128):
            blocks += [
                nn.Conv2d(width, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            width = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.pool(self.features(x)).flatten(1)))


def build_audio(model_id: str, num_classes: int, *, pretrained: bool, freeze_backbone: bool = False) -> nn.Module:
    """A spectrogram classifier. `audio_resnet18` is torchvision's ResNet-18 with its first convolution turned into a
    one-channel one by summing the pretrained filters over red, green and blue, so ImageNet knowledge still applies."""
    if model_id not in tasks.AUDIO_MODELS_BY_ID:
        raise ValueError(f'"{model_id}" is not an audio model')
    if model_id == "audio_cnn":
        net: nn.Module = AudioCnn(num_classes)
        head = net.fc
    else:
        resnet = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        first = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            first.weight.copy_(resnet.conv1.weight.sum(dim=1, keepdim=True))
        resnet.conv1 = first
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        net, head = resnet, resnet.fc
    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return net

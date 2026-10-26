"""Hugging Face classifiers (images and texts) as a network the rest of the backend can train, serve and export.

Imports torch (and transformers inside functions): only loaded where a model is really built. Nothing here ever trusts
remote code or unpickles weights: local files only, safetensors only.

The backend's tensors are what the exported ONNX graph takes: `[N, 3, H, W]` floats for an image, `[N, max_tokens]`
int64 ids for a text. transformers wants them under names (`pixel_values`, `input_ids` and an attention mask) and
answers with a `ModelOutput`. `HFImageNet` and `HFTextNet` are that one translation and a `.logits`, so everything else
(the training loop, ONNX export, serving) treats a Hugging Face model like any other network. A text network takes ONLY
the ids and derives its attention mask from the padding id, so the exported graph has one input and a client has one
thing to make.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn

# The classification head of every supported family: a `Linear`, a `Flatten` then a `Linear` (ResNet), or a small
# module that ends in an `out_proj` `Linear` (RoBERTa, ELECTRA).
_HEAD = "classifier"
_HEAD_PREFIXES = (f"{_HEAD}.", "pre_classifier.")
# What a published tokenizer is made of: the fast tokenizer's own file, or the vocabulary files it is converted from.
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "vocab.json", "spiece.model", "sentencepiece.bpe.model")


class HFImageNet(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=images).logits


class HFTextNet(nn.Module):
    def __init__(self, model: nn.Module, pad_id: int) -> None:
        super().__init__()
        self.model = model
        self.pad_id = pad_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=(input_ids != self.pad_id).long()).logits


def _options() -> dict[str, object]:
    return {"local_files_only": True, "use_safetensors": True, "trust_remote_code": False}


def _is_head(key: object, *, text: bool) -> bool:
    """Whether a missing weight is one that is meant to be made up: the classification head, and for a text model the
    pooler that a checkpoint published for masked language modelling does not carry."""
    name = str(key)
    return name.startswith(_HEAD_PREFIXES) or (text and ".pooler." in f".{name}")


def _outside_head(keys: Any, *, text: bool) -> list[str]:
    return sorted(str(k) for k in keys if not _is_head(k, text=text))


def _load_checkpoint(path: str, *, text: bool) -> nn.Module:
    """The checkpoint at `path` as a classifier, with every weight outside the head taken from the files.

    A trained classifier loads as it is. A bare backbone (the usual way a base model is published) gets a new head,
    which is the one thing that is meant to be missing. `from_pretrained` alone is not enough to know that: it
    re-initialises anything it cannot find and only says so in a warning, and for a vision transformer it does not
    always connect a bare backbone's keys to the classifier's. So the backbone is loaded on its own, strictly, and
    copied in, and any tensor still unaccounted for is an error: the alternative is a model that trains from random
    weights while claiming to be pretrained."""
    from transformers import AutoModel, AutoModelForImageClassification, AutoModelForSequenceClassification

    classifier_class = AutoModelForSequenceClassification if text else AutoModelForImageClassification
    model, info = classifier_class.from_pretrained(path, output_loading_info=True, **_options())
    if info.get("mismatched_keys"):
        raise ValueError(f"the weights do not match the config ({sorted(info['mismatched_keys'])[0]})")
    if not _outside_head(info.get("missing_keys", ()), text=text):
        return model

    backbone, backbone_info = AutoModel.from_pretrained(path, output_loading_info=True, **_options())
    absent = _outside_head(backbone_info.get("missing_keys", ()), text=text)
    if absent:
        raise ValueError(f"the checkpoint has no weights for {absent[0]}")
    result = getattr(model, model.base_model_prefix).load_state_dict(backbone.state_dict(), strict=False)
    absent = _outside_head(result.missing_keys, text=text)
    if absent:
        raise ValueError(f"the checkpoint has no weights for {absent[0]}")
    return model


def _replace_head(model: nn.Module, num_classes: int) -> None:
    head = getattr(model, _HEAD, None)
    if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
        parent, name, old = head, str(len(head) - 1), head[-1]
    elif isinstance(head, nn.Linear):
        parent, name, old = model, _HEAD, head
    elif head is not None and isinstance(getattr(head, "out_proj", None), nn.Linear):
        parent, name, old = head, "out_proj", head.out_proj
    else:
        raise ValueError("it has no linear classification head to replace")
    new = nn.Linear(old.in_features, num_classes)
    config = model.config
    nn.init.normal_(new.weight, std=getattr(config, "initializer_range", 0.02))
    nn.init.zeros_(new.bias)
    if name.isdigit():
        parent[int(name)] = new  # type: ignore[index]
    else:
        setattr(parent, name, new)


def _name_the_classes(model: nn.Module, names: list[str]) -> None:
    config = model.config
    model.num_labels = config.num_labels = len(names)
    config.id2label = dict(enumerate(names))
    config.label2id = {name: i for i, name in enumerate(names)}


def build(path: str, names: list[str]) -> HFImageNet:
    """The image checkpoint at `path` with a fresh `len(names)`-way classification head."""
    model = _load_checkpoint(path, text=False)
    _replace_head(model, len(names))
    _name_the_classes(model, names)
    return HFImageNet(model)


def build_text(path: str, names: list[str], pad_id: int) -> HFTextNet:
    """The text checkpoint at `path` with a fresh `len(names)`-way classification head. Texts pad with `pad_id`."""
    model = _load_checkpoint(path, text=True)
    own = getattr(model.config, "pad_token_id", None)
    if own is not None and own != pad_id:
        # RoBERTa numbers positions from the padding id it was built with, so a tokenizer that pads with another id
        # would shift every position and the model would quietly read nonsense.
        raise ValueError(f"its tokenizer pads with id {pad_id} but the model expects {own}")
    _replace_head(model, len(names))
    _name_the_classes(model, names)
    return HFTextNet(model, pad_id)


def freeze_backbone(net: nn.Module) -> None:
    """Train only the head: its layers, and for a text model the pooler in front of it."""
    for name, param in net.named_parameters():
        inside = name.removeprefix("model.")
        param.requires_grad = inside.startswith(_HEAD_PREFIXES) or (isinstance(net, HFTextNet) and ".pooler." in name)


def save(net: nn.Module, dest: str) -> None:
    """A self-contained folder (config.json plus model.safetensors) `load` can read with no other file."""
    assert isinstance(net, (HFImageNet, HFTextNet))
    net.model.save_pretrained(dest, safe_serialization=True)


def _load_trained(path: str, *, text: bool) -> nn.Module:
    from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification

    classifier_class = AutoModelForSequenceClassification if text else AutoModelForImageClassification
    model, info = classifier_class.from_pretrained(path, output_loading_info=True, **_options())
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError("the trained model's files are incomplete")
    return model


def load(path: str) -> HFImageNet:
    """A trained image model written by `save`. Every tensor must be in the file: a trained model has no head to
    make up."""
    return HFImageNet(_load_trained(path, text=False))


def load_text(path: str, pad_id: int) -> HFTextNet:
    """A trained text model written by `save`."""
    return HFTextNet(_load_trained(path, text=True), pad_id)


def smoke_test(path: str) -> None:
    """Build the model with a two-class head and push one blank image through it. Raises if anything is wrong."""
    from omnitrain.plugins.backends.pytorch.custom import image_settings

    size = image_settings(Path(path))["imageSize"]
    net = build(path, ["a", "b"]).eval()
    with torch.no_grad():
        logits = net(torch.zeros(1, 3, size, size))
    if tuple(logits.shape) != (1, 2):
        raise ValueError(f"expected one score per class, got shape {tuple(logits.shape)}")


# -- tokenizer -------------------------------------------------------------------------------------------------------


def read_tokenizer(path: str) -> tuple[str, int, str]:
    """(the tokenizer as the JSON string `tokenizers` writes, its padding id, its padding token) for the tokenizer
    files in `path`. Only a fast (Rust) tokenizer is accepted: it is a data file that every Hugging Face `tokenizers`
    binding runs the same way, which is what lets a client reproduce it. A model that only has a Python tokenizer is
    refused."""
    from transformers import AutoTokenizer

    # Without files to read, transformers does not fail: it makes a tokenizer with an empty vocabulary, which would turn
    # every word into the unknown token and train a model on nothing.
    if not any((Path(path) / name).is_file() for name in TOKENIZER_FILES):
        raise ValueError("it has no tokenizer files (tokenizer.json, vocab.txt, vocab.json or a sentencepiece model)")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("it has no fast tokenizer (tokenizer.json)")
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise ValueError("its tokenizer has no padding token")
    return tokenizer.backend_tokenizer.to_str(), int(tokenizer.pad_token_id), str(tokenizer.pad_token)


def make_tokenizer(spec: Any) -> Any:
    """A `tokenizers.Tokenizer` for `spec` (an `HFTextSpec`) that cuts at `max_tokens` and pads to it: exactly the
    calls the Python client makes."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(spec.tokenizer)
    tokenizer.enable_truncation(spec.max_tokens)
    tokenizer.enable_padding(length=spec.max_tokens, pad_id=spec.pad_id, pad_token=spec.pad_token)
    return tokenizer


def encode_texts(tokenizer: Any, texts: list[str]) -> torch.Tensor:
    """`[rows, max_tokens]` int64 ids."""
    if not texts:
        return torch.zeros((0, 0), dtype=torch.long)
    return torch.tensor([e.ids for e in tokenizer.encode_batch(texts)], dtype=torch.long)


def smoke_test_text(path: str) -> None:
    """Read the tokenizer, build the model with a two-class head and classify one short text. Raises if anything is
    wrong."""
    from omnitrain.plugins.backends.pytorch.custom import text_token_limit
    from omnitrain.plugins.backends.pytorch.spec import HFTextSpec

    tokenizer_json, pad_id, pad_token = read_tokenizer(path)
    spec = HFTextSpec(
        max_tokens=min(16, text_token_limit(path)), pad_id=pad_id, pad_token=pad_token, tokenizer=tokenizer_json
    )
    ids = encode_texts(make_tokenizer(spec), ["a short text to classify"])
    net = build_text(path, ["a", "b"], pad_id).eval()
    with torch.no_grad():
        logits = net(ids)
    if tuple(logits.shape) != (1, 2):
        raise ValueError(f"expected one score per class, got shape {tuple(logits.shape)}")

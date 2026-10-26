"""The preprocessing.json that makes a bare PyTorch ONNX export usable.

An image graph takes a ready-made float tensor and returns raw class logits: no decoding of an image file, no resize,
no label lookup. A sound or a text is the same: the graph takes the spectrogram or the word ids, and the manifest
carries the whole recipe that makes them. A table graph takes raw numbers (it fills and standardises them itself, so a
client does nothing to them) and returns logits or the target's number. `names.json` (written at train time) supplies
what a client must do to an image first and the class list in the model's own index order. The manifest is built from
the same spec the model was trained, evaluated and served with, so a client that follows it sees what the platform
saw.

The class list MUST come from `names.json`, NEVER from the label_classes table: Postgres has no idea which index a run
assigned each class, so guessing wrong silently mislabels every prediction a generated client makes.
"""

from typing import Any

from omnitrain.domain.token_tags import OUTSIDE
from omnitrain.plugins.backends.pytorch import audio, text
from omnitrain.plugins.backends.pytorch.runtime import METADATA_FILENAME
from omnitrain.plugins.backends.pytorch.spec import AudioSpec, HFTextSpec, ImageSpec, TextSpec, WordsSpec


def build_preprocessing_manifest(run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: turn the training metadata into the preprocessing manifest."""
    if meta is None:
        raise ValueError(f"Run {run_id} has no {METADATA_FILENAME} to build a preprocessing manifest from")

    if meta["task"] == "classify":
        output: dict[str, Any] = {
            "name": meta["outputName"],
            "type": "category",
            "column": meta["outputName"],
            "classes": list(meta["names"]),
        }
    elif meta["task"] == "tag":
        # One row of logits per word position, `[batch, maxTokens, len(classes)]`; `classes` is the tag order.
        output = {
            "name": meta["outputName"],
            "type": "tags",
            "column": meta["outputName"],
            "classes": list(meta["names"]),
        }
    else:
        output = {"name": meta["outputName"], "type": "number", "column": meta["outputName"]}

    if meta["inputKind"] == "tabular":
        # One float input per column; a missing value is NaN (or simply absent), which the graph fills in. There is
        # deliberately no `numberNormalization`: the graph standardises, and a client doing it too would do it twice.
        inputs: list[dict[str, Any]] = [
            {"name": column, "type": "number", "column": column} for column in meta["inputColumns"]
        ]
        return {"schemaVersion": 1, "runId": run_id, "inputs": inputs, "outputs": [output]}

    if meta["inputKind"] == "audio":
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "inputs": [_audio_input(meta)],
            "outputs": [output],
        }

    if meta["inputKind"] == "words":
        return {"schemaVersion": 1, "runId": run_id, "inputs": [_words_input(meta)], "outputs": [output]}

    if meta["inputKind"] == "text":
        return {"schemaVersion": 1, "runId": run_id, "inputs": [_text_input(meta)], "outputs": [output]}

    spec = ImageSpec.from_meta(meta)
    image = {
        "name": "images",
        "type": "image",
        "column": meta["imageColumn"],
        # `images.preprocess`: RGB, resize the SHORTER edge to `size` (bilinear, keeping the aspect ratio), then
        # centre-crop a size x size square. Every generated client implements this mode; without an `imageResize`
        # they squash the whole image to height x width, which is what Ludwig does.
        "imageResize": {"mode": "shorter_edge_center_crop", "interpolation": "bilinear"},
        # Scaled to 0..1 first, then (x - mean) / std per channel.
        "imageNormalization": {"mean": list(spec.mean), "std": list(spec.std)},
        # The key the generated clients read the target size from (they fall back to 224 without it).
        "ludwigPreprocessing": {"height": spec.size, "width": spec.size},
    }
    return {"schemaVersion": 1, "runId": run_id, "inputs": [image], "outputs": [output]}


TEXT_SPLIT = (
    "Lower-case the ASCII letters A-Z only (add 32 to a code unit between 65 and 90) and change nothing else. A "
    "token is a maximal run of code units none of which is in separatorRanges; the text is read as UTF-16 code units "
    "or as code points, which gives the same tokens because every separator is in the Basic Multilingual Plane. No "
    "Unicode normalisation, no other case folding."
)
TEXT_IDS = (
    "Take the first maxTokens tokens. Each becomes its position in vocabulary plus firstWordId, or unknownId when "
    "it is not in vocabulary (an exact, case-sensitive match on the lower-cased token). Pad the end with padId up to "
    "maxTokens. A text with no tokens is all padding."
)


def _text_input(meta: dict[str, Any]) -> dict[str, Any]:
    """The whole recipe that turns a text into the model's input; `text.py` is the reference implementation and every
    generated client follows this object (it refuses a `recipe` it does not know). Per text: split into tokens, turn
    them into ids, run the model on the one row, softmax the logits."""
    if "hfTokenizer" in meta:
        return _hf_text_input(meta)
    spec = TextSpec.from_meta(meta)
    return {
        "name": "tokens",
        "type": "text",
        "column": meta["textColumn"],
        "layout": "[batch, maxTokens] int64",
        "textTokens": {
            "recipe": "word_ids_v1",
            "split": TEXT_SPLIT,
            "lowercase": {"from": ord("A"), "to": ord("Z"), "add": 32},
            "separatorRanges": [[lo, hi] for lo, hi in text.SEPARATOR_RANGES],
            "ids": TEXT_IDS,
            "maxTokens": spec.max_tokens,
            "padId": text.PAD,
            "unknownId": text.UNKNOWN,
            "firstWordId": text.FIRST_WORD,
            "vocabulary": list(spec.vocab),
        },
    }


TAG_WORDS_SPLIT = (
    "A word is a maximal run of characters that are not whitespace, whitespace being what Python's str.split() "
    "separates on (str.isspace()), so punctuation stays attached to its word. Lower-case the ASCII letters A-Z only "
    "(add 32 to a code unit between 65 and 90) and change nothing else."
)
TAG_WORDS_IDS = (
    "Take the first maxTokens words. Each becomes its position in vocabulary plus firstWordId, or unknownId when it is "
    "not in vocabulary (an exact match on the lower-cased word). Pad the end with padId up to maxTokens. The model "
    "returns one row of tag logits per position: the tag of word i is the highest of row i, for i below the number of "
    "words read. A word past maxTokens was not read and gets fallbackTag."
)


def _words_input(meta: dict[str, Any]) -> dict[str, Any]:
    """What turns a text into a tagger's input: a word per id (`tagging.py` is the reference implementation). Every
    generated client that follows it refuses a `recipe` it does not know."""
    spec = WordsSpec.from_meta(meta)
    names = list(meta["names"])
    return {
        "name": "tokens",
        "type": "text",
        "column": meta["textColumn"],
        "layout": "[batch, maxTokens] int64",
        "tagWords": {
            "recipe": "tag_word_ids_v1",
            "split": TAG_WORDS_SPLIT,
            "lowercase": {"from": ord("A"), "to": ord("Z"), "add": 32},
            "ids": TAG_WORDS_IDS,
            "maxTokens": spec.max_tokens,
            "padId": text.PAD,
            "unknownId": text.UNKNOWN,
            "firstWordId": text.FIRST_WORD,
            "fallbackTag": OUTSIDE if OUTSIDE in names else names[0],
            "vocabulary": list(spec.vocab),
        },
    }


HF_TOKENIZER_RUN = (
    "Load tokenizer with the Hugging Face tokenizers library (Tokenizer.from_str, or tokenizer.json in a bundle). Call "
    "enable_truncation(maxTokens), then enable_padding(length=maxTokens, pad_id=padId, pad_token=padToken), then "
    "encode the text; the ids are the model input. The special tokens the tokenizer adds count towards maxTokens. A "
    "missing text is the empty string."
)


def _hf_text_input(meta: dict[str, Any]) -> dict[str, Any]:
    """A bring-your-own transformer's input: the ids its own tokenizer makes. The tokenizer is data that every Hugging
    Face `tokenizers` binding runs the same way, so the manifest carries it whole (`tokenizer` is the JSON string that
    library writes) and a client that has the library needs nothing else."""
    spec = HFTextSpec.from_meta(meta)
    return {
        "name": "input_ids",
        "type": "text",
        "column": meta["textColumn"],
        "layout": "[batch, maxTokens] int64",
        "hfTokenizer": {
            "recipe": "hf_tokenizer_v1",
            "run": HF_TOKENIZER_RUN,
            "maxTokens": spec.max_tokens,
            "padId": spec.pad_id,
            "padToken": spec.pad_token,
            "tokenizer": spec.tokenizer,
        },
    }


RESAMPLE = (
    "Always applied, even for equal rates (then the identity). g = gcd(source, target), up = target / g, "
    "down = source / g, big = max(up, down), H = 10 * big. h[n] = up * w[n] * s[n] / S for n in 0..2H, where "
    "s[n] = sinc((n - H) / big) / big, sinc(x) = sin(pi x) / (pi x), w is a symmetric Kaiser window of 2H + 1 points "
    "(beta = 5) and S = sum of w[n] * s[n]. The output has ceil(len * up / down) samples: out[i] = sum over k in -H..H "
    "of h[H + k] * x[(i * down + k) / up], counting only the k for which i * down + k is divisible by up and the index "
    "is inside the recording. This is scipy.signal.resample_poly(x, up, down)."
)
WINDOW_STARTS = (
    "room = length - windowSamples. A recording no longer than one window gives one window at 0, zero-padded at the "
    "end. A longer one gives, for windowsPerAudio > 1, the distinct values of round(i * room / (windowsPerAudio - 1)) "
    "for i in 0..windowsPerAudio-1 (round half to even, ascending); for 1, room // 2."
)
MEL_FORMULA = (
    "mel = 2595 * log10(1 + hz / 700). nFft / 2 + 1 bins spaced evenly from 0 Hz to sampleRate / 2; bands + 2 edge "
    "points evenly spaced in mel from fMin to fMax; band j is the triangle rising from edge j to edge j + 1, falling "
    "to edge j + 2, with height 1 at edge j + 1 (no area normalisation)."
)


def _audio_input(meta: dict[str, Any]) -> dict[str, Any]:
    """The whole recipe that turns an audio file into the model's input; `audio.py` is the reference implementation and
    every generated client follows this object (it refuses a `recipe` it does not know). The numbers are read by the
    clients, the strings say in words what they do. Per recording: decode, average the channels, resample, take
    `windowsPerAudio` windows, make a spectrogram of each, run the model on them, softmax each window's logits and
    average the probabilities."""
    spec = AudioSpec.from_meta(meta)
    return {
        "name": "spectrograms",
        "type": "audio",
        "column": meta["audioColumn"],
        "layout": "[batch, 1, nMels, frames]",
        "audioSpectrogram": {
            "recipe": "log_mel_v1",
            "decode": "WAV: 8-bit unsigned (v - 128) / 128, signed 16-bit v / 32767, 24-bit v / 8388607, 32-bit v / "
            "2147483647, 32/64-bit float as is. Other containers only where the client says so",
            "channels": "mean of all channels",
            "sampleRate": spec.sample_rate,
            "resample": {
                "method": "kaiser_polyphase",
                "kaiserBeta": 5.0,
                "halfLengthPerRate": 10,
                "definition": RESAMPLE,
            },
            "windowSeconds": spec.seconds,
            "windowSamples": spec.length,
            "windowStarts": WINDOW_STARTS,
            "windowsPerAudio": spec.clips,
            "stft": {
                "nFft": spec.n_fft,
                "hopLength": spec.hop,
                "window": "hann (periodic: 0.5 - 0.5 * cos(2 * pi * n / nFft))",
                "center": False,
                "power": 2,
            },
            "mel": {
                "scale": "htk",
                "bands": spec.n_mels,
                "fMin": spec.f_min,
                "fMax": spec.f_max,
                "norm": "none",
                "formula": MEL_FORMULA,
            },
            "logFloor": audio.LOG_FLOOR,
            "log": "ln(mel + logFloor)",
            "normalization": {"mean": spec.mean, "std": spec.std},
            "frames": spec.frames,
        },
    }

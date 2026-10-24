# Preprocessing ops

Deterministic transforms applied when a snapshot is created (never at training time). Each op is scoped to the
splits (train, validation, test) the snapshot config selects for it. A preprocessed item **replaces** its original
in the snapshot's membership for exactly those splits; nothing is added. Random, train-only extra copies are
[augmentation](../augmentation/README.md), a separate package. The ops live in [`ops/`](ops/).

Adding an op needs no frontend change: `GET /api/projects/{id}/preprocessing` lists it and the snapshot dialog
renders its parameters from the `Params` model.

## How an op is found

`Registry` imports every module directly under `omnitrain.plugins.preprocessing.ops` whose name does not start
with `_`. A `Preprocessing` subclass registers itself when created, but only if it sets its **own** `id`. A
duplicate `id` raises at import time.

## Adding an op

1. **Put it in a module** under `ops/`: an existing one for its modality (`vision.py`, `text.py`, `audio.py`,
   `video.py`, `tabular.py`) or a new `ops/<name>.py`.
2. **Subclass `Preprocessing`** and set:
   - `id`, a stable unique string. It is stored in the snapshot config, so do not rename it later.
   - `label`, and optionally `description`.
   - `modality`: `"vision"`, `"text"`, `"audio"`, `"video"` or `"tabular"`. The list is closed.
   - `order`, the sort key inside a modality.
   - `Params`, a pydantic model if the op has settings. Subclass `ParamsModel` (camelCase on the wire, unknown
     keys rejected); the default `NoParams` means no settings.
3. **Implement `apply(sample, params, state=None)`**, a classmethod that returns a **new** sample.
   Never mutate the input, and use no randomness.
4. **If the op needs dataset statistics**, implement `fit(samples)`. It runs once on the **train** split only and
   its result arrives as `state` in `apply`. That is how nothing from validation or test leaks in.
5. **Override `supports(task)`** only if the default is wrong. The default offers an op for classification-style
   tasks (`annotation.type == "classification"`) of the same modality that are not `planned`. A transform that
   rewrites the input cannot keep boxes or token tags aligned, so those tasks get no ops unless you opt in.
6. **Test** it like `server/tests/test_preprocessing.py`.

Samples are plain objects per modality:

| Modality | Sample |
| --- | --- |
| vision | `PIL.Image.Image` |
| text | `str` |
| audio | `AudioClip` |
| video | `VideoClip` (all frames decoded, uint8 RGB) |
| tabular | `dict[str, Any]`, the item's `features_json` |

## Minimal example

```python
# ops/text.py
from typing import Any

from pydantic import Field

from omnitrain.plugins.preprocessing.base import ParamsModel, Preprocessing


class TruncateWords(Preprocessing):
    class Params(ParamsModel):
        max_words: int = Field(256, ge=1, le=4096, title="Max words", description="Keep at most this many words.")

    id = "text_truncate_words"
    label = "Truncate to N words"
    description = "Cut long texts after a number of words."
    modality = "text"
    order = 30

    @classmethod
    def apply(cls, sample: str, params: Params, state: Any = None) -> str:
        return " ".join(sample.split()[: params.max_words])
```

An op with nothing to tune is the same without `Params` (see `Lowercase` in `ops/text.py`). For `fit`, see
`Standardize` in `ops/tabular.py`.

## Rules that are easy to break

- **An op with nothing to do must return the sample it was given** (the same object). The services read
  `new is sample` as "leave the original alone" and skip a lossy re-encode. This matters most for video, which is
  always re-encoded as VP8 WebM.
- **Derived items are real `dataset_items` rows** with `source_item_id` pointing at the pool ancestor, with files
  under `snapshots/{versionId}/preprocessed/`. You do not write them; `services/preprocessing.py` does. Do not
  assume a chain of derived items.
- **Preprocessing is not replayed at inference or export time.** A trained model only ever sees what the snapshot
  froze. Do not rely on an op running again later.
- **Keep file modalities cheap in `fit`.** Only text and tabular receive samples in `fit`; for vision and audio the
  list is empty because loading every file would not scale.
- **Admin switches only filter** `list_preprocessing` and the validation of a new config. Existing snapshots keep
  working when an op is switched off.

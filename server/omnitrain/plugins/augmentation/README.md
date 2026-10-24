# Augmentation ops

Random transforms applied when a snapshot is created (never at training time). They are **train-split only**:
`services/augmentation.py` adds N extra copies of each train item, made from whatever the train split holds after
[preprocessing](../preprocessing/README.md) ran, so a copy carries forward any `preprocessing` its source had.
Validation and test are never augmented. The ops live in [`ops/`](ops/).

Adding an op needs no frontend change: `GET /api/projects/{id}/augmentations` lists it and the snapshot dialog
renders its parameters from the `Params` model.

## How an op is found

`Registry` imports every module directly under `omnitrain.plugins.augmentation.ops` whose name does not start
with `_`. An `Augmentation` subclass registers itself when created, but only if it sets its **own** `id`. A
duplicate `id` raises at import time. Shared helpers belong outside `ops/` (like `augmentation/media.py`) or in an
underscore-prefixed module.

## Adding an op

1. **Put it in a module** under `ops/`: an existing one for its modality (`vision.py`, `text.py`, `audio.py`,
   `video.py`, `tabular.py`) or a new `ops/<name>.py`.
2. **Subclass `Augmentation`** and set:
   - `id`, a stable unique string. It is stored in the snapshot config, so do not rename it later.
   - `label`, and optionally `description`.
   - `modality`: `"vision"`, `"text"`, `"audio"`, `"video"` or `"tabular"`. The list is closed.
   - `order`, the sort key inside a modality.
   - `Params`, a nested pydantic model if the op has settings. Subclass `ParamsModel` (camelCase on the wire,
     unknown keys rejected). The default `NoParams` means no settings.
3. **Implement `apply(sample, params, rng, state=None)`**, a classmethod that returns a **new** sample. Never
   mutate the input. All randomness comes from the `rng` you are given (a `random.Random`), never the global
   `random`, so a seeded snapshot reproduces.
4. **If the op needs dataset statistics**, implement `prepare(samples)`. It runs once on the originals and the
   result arrives as `state`. Only text and tabular get samples; for vision and audio the list is empty.
5. **Override `supports(task)`** only if the default is wrong. The default offers an op for classification-style
   tasks (`annotation.type == "classification"`) of the same modality that are not `planned`. A transform that
   rewrites the input cannot keep boxes or token tags aligned, so those tasks get no ops unless you opt in.
6. **Test** it like `server/tests/test_augmentation.py`.

Samples are plain objects per modality: vision `PIL.Image.Image`, text `str`, audio `AudioClip`, video
`VideoClip`, tabular `dict[str, Any]` (the item's `features_json`).

## Minimal example

```python
# ops/text.py
import random
from typing import Any

from pydantic import Field

from omnitrain.plugins.augmentation.base import Augmentation, ParamsModel


class RandomWordShuffleTail(Augmentation):
    class Params(ParamsModel):
        keep_words: int = Field(3, ge=1, le=50, title="Words kept in place", description="Leading words left alone.")

    id = "text_shuffle_tail"
    label = "Shuffle the tail"
    description = "Keep the first few words and shuffle the rest."
    modality = "text"
    order = 50

    @classmethod
    def apply(cls, sample: str, params: Params, rng: random.Random, state: Any = None) -> str:
        words = sample.split()
        head, tail = words[: params.keep_words], words[params.keep_words :]
        rng.shuffle(tail)
        return " ".join(head + tail)
```

Look at `ops/text.py` for four complete ops, including `json_schema_extra={"step": ...}` to set a slider step.

## Rules that are easy to break

- **An op with nothing to do must return the sample it was given** (the same object). The services read
  `new is sample` as "leave the original alone" and skip a lossy re-encode.
- **Augmented items are real `dataset_items` rows** with `augmentation` set and `source_item_id` pointing at the
  pool ancestor (never at an intermediate row). Their files live under `snapshots/{versionId}/augmented/` and are
  deleted with the snapshot. `services/augmentation.py` writes them, not the op.
- **Video**: derived clips are always written as VP8 WebM, and a clip is fully decoded in memory, so keep ops
  frugal.
- **Augmentation is not replayed at inference or export time.** The model only sees what the snapshot froze.
- **Admin switches only filter** `list_augmentations` and the validation of a new config.

# Headline scores

A run's headline score is the single number the runs list, the "best run" marker, the overview and the comparison
show. It is not always accuracy: `services/evaluation_score.py` picks it from the evaluation report's `overall`
block (accuracy for a classifier, mAP50 for a detector, R² for a regressor) and ships it as `evaluation.score`
(`metric`, `label`, `value`, `higherIsBetter`) on every run summary. One `HeadlineScore` subclass per metric key.

Adding one needs no frontend edit: the web reads the label and direction off `evaluation.score` through
`web/lib/runScore.ts`. Do not hard-code `accuracy` anywhere new.

## How a score is found

`Registry` imports every module directly under `omnitrain.scoring` whose name does not start with `_`
(`metrics.py` holds the built-ins). A `HeadlineScore` subclass registers itself when created, but only if it sets its
**own** `id`. It must also set a `label`, or class creation raises `TypeError`.

## Adding a score

1. **First make a backend report the metric.** The score reads the `overall` block of the report that
   `TrainerBackend.evaluate()` returns. If no backend writes your key there, nothing will ever match.
2. **Add a class** in `metrics.py` (or a new module here):
   - `id`, the key in `overall`.
   - `label`, shown in the UI.
   - `higher_is_better = False` for an error metric (default `True`).
   - `order`, the priority. When a report carries several known metrics, the **lowest `order` wins**.
3. **Override `value(overall)`** only if the metric is stored in a different shape. The default reads a plain
   number at `id` and ignores booleans and missing keys.
4. **Test** it like `server/tests/test_evaluation_score.py`.

## Minimal example

```python
# metrics.py
class F1Macro(HeadlineScore):
    id = "f1_macro"
    label = "Macro F1"
    order = 15  # after accuracy (10), before mAP50 (20)
```

Pick `order` carefully: a report that has both `accuracy` and your metric shows whichever has the lower `order`.
The built-ins are `accuracy` 10, `map50` 20, `r2` 30, `rmse` 40, `mae` 50.

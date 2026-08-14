from typing import Any


def parse_prediction_row(
    feature_name: str,
    feature_type: str,
    predictions: Any,
    idx2str: list[str] | None,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Parse row 0 of a Ludwig prediction DataFrame into a confidence dict.

    For `category` outputs this is `{class_name: probability}`, filtered to
    `>= threshold` and sorted descending. For `number` outputs it's the
    single `{feature_name: value}` pair. Falls back to the raw
    `{feature}_predictions` column when no probability column is present.
    """
    prob_col = f"{feature_name}_probabilities"
    pred_col = f"{feature_name}_predictions"

    results: dict[str, float] = {}
    if feature_type == "category" and prob_col in predictions.columns:
        probs = predictions[prob_col].iloc[0]
        if idx2str and isinstance(probs, (list, tuple)):
            for idx, prob in enumerate(probs):
                if prob >= threshold and idx < len(idx2str):
                    results[idx2str[idx]] = round(float(prob), 4)
        elif isinstance(probs, dict):
            for class_name, prob in probs.items():
                if prob >= threshold:
                    results[str(class_name)] = round(float(prob), 4)

    if not results and pred_col in predictions.columns:
        predicted = predictions[pred_col].iloc[0]
        if feature_type == "number":
            results[feature_name] = round(float(predicted), 4)
        else:
            results[str(predicted)] = 1.0

    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

from typing import Any, Literal, TypedDict

import pandas as pd


class ClassificationClass(TypedDict):
    label: str
    confidence: float


class ClassificationOutput(TypedDict):
    kind: Literal["classification"]
    feature: str
    classes: list[ClassificationClass]


class RegressionOutput(TypedDict):
    kind: Literal["regression"]
    feature: str
    value: float


class TextOutput(TypedDict):
    kind: Literal["text"]
    feature: str
    text: str


class TokenTag(TypedDict):
    token: str
    tag: str


class TokensOutput(TypedDict):
    kind: Literal["tokens"]
    feature: str
    tokens: list[TokenTag]


InferenceOutput = ClassificationOutput | RegressionOutput | TextOutput | TokensOutput


def build_inference_output(
    feature_name: str,
    feature_type: str,
    predictions: Any,
    idx2str: list[str] | None,
    top_k: int = 100,
    input_tokens: list[str] | None = None,
) -> InferenceOutput:
    """Turn row 0 of a Ludwig prediction DataFrame into the tagged-union
    shape `InferenceResponseSchema.output` expects (see server/lib/schema.ts).

    Distinct from `parse_prediction_row` below, which is kept as-is for the
    export golden-sample path — every devkit/app template consumes
    `expected.json`'s `predictions` field as a flat `{class: probability}`
    dict, so changing its shape would break every generated client. This
    function instead represents every output type an implemented task can
    actually produce: classification, regression, generated text, and
    token-tagged sequences.
    """
    prob_col = f"{feature_name}_probabilities"
    pred_col = f"{feature_name}_predictions"

    if feature_type == "category":
        classes: list[ClassificationClass] = []
        if prob_col in predictions.columns:
            probs = predictions[prob_col].iloc[0]
            if idx2str and isinstance(probs, (list, tuple)):
                classes = [
                    {"label": idx2str[idx], "confidence": round(float(p), 4)}
                    for idx, p in enumerate(probs)
                    if idx < len(idx2str)
                ]
            elif isinstance(probs, dict):
                classes = [{"label": str(k), "confidence": round(float(v), 4)} for k, v in probs.items()]
        if not classes and pred_col in predictions.columns:
            classes = [{"label": str(predictions[pred_col].iloc[0]), "confidence": 1.0}]
        classes.sort(key=lambda c: c["confidence"], reverse=True)
        return {"kind": "classification", "feature": feature_name, "classes": classes[:top_k]}

    if feature_type == "number":
        value = float(predictions[pred_col].iloc[0]) if pred_col in predictions.columns else 0.0
        return {"kind": "regression", "feature": feature_name, "value": round(value, 4)}

    if feature_type == "sequence":
        predicted = predictions[pred_col].iloc[0] if pred_col in predictions.columns else []
        predicted_tags = list(predicted) if isinstance(predicted, (list, tuple)) else [str(predicted)]
        # Best-effort alignment with the input tokens: Ludwig's tagger
        # output is already one tag per input token for the common
        # (space-tokenized) case, but if the lengths disagree — sub-word
        # tokenization we can't invert here — pad/truncate rather than
        # raise. A slightly misaligned tag list is far more useful to a
        # caller than a 502.
        tokens = input_tokens or []
        if tokens:
            paired = list(zip(tokens, predicted_tags))
        else:
            paired = [("", tag) for tag in predicted_tags]
        if len(paired) < len(predicted_tags):
            paired += [("", tag) for tag in predicted_tags[len(paired) :]]
        return {
            "kind": "tokens",
            "feature": feature_name,
            "tokens": [{"token": t, "tag": str(tag)} for t, tag in paired],
        }

    # Generated text — `text_generation` / `summarization` /
    # `sequence_to_sequence` / `question_answering` (all `modelType: 'llm'`)
    # output feature type is `text`. Also the fallback for any output type
    # not covered above, so an unanticipated Ludwig feature type degrades
    # to a stringified value instead of raising.
    text = str(predictions[pred_col].iloc[0]) if pred_col in predictions.columns else ""
    return {"kind": "text", "feature": feature_name, "text": text}


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


def build_batch_result_frame(input_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Concatenate a batch inference job's original input rows with Ludwig's
    own postprocessed prediction columns (`{feature}_predictions`,
    `{feature}_probability`, ...) — the same "input + Ludwig's raw
    prediction columns" shape as the evaluation predictions.parquet (see
    ai_service/services/evaluate.py), so a batch result and an evaluation
    export read the same way. Both frames come from the same `model.predict`
    call and are therefore already row-aligned; the index reset just avoids
    a spurious join on a non-default input index.
    """
    return pd.concat([input_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)

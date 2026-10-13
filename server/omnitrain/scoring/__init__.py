"""Headline score plugins: one class per metric that can say how good a run is. See `base.HeadlineScore`."""

from omnitrain.scoring.base import HeadlineScore, find_score, list_scores

__all__ = ["HeadlineScore", "find_score", "list_scores"]

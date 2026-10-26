"""The built-in headline scores. Order is the priority a report's metrics are tried in."""

from omnitrain.scoring.base import HeadlineScore


class EntityF1(HeadlineScore):
    """Tagging with BIO tags: whole entities exactly right. Before accuracy, which `O` words inflate."""

    id = "entityF1"
    label = "Entity F1"
    order = 5


class Accuracy(HeadlineScore):
    id = "accuracy"
    label = "Accuracy"
    order = 10


class MeanAveragePrecision50(HeadlineScore):
    id = "map50"
    label = "mAP50"
    order = 20


class R2(HeadlineScore):
    id = "r2"
    label = "R²"
    order = 30


class RootMeanSquaredError(HeadlineScore):
    id = "rmse"
    label = "RMSE"
    higher_is_better = False
    order = 40


class MeanAbsoluteError(HeadlineScore):
    id = "mae"
    label = "MAE"
    higher_is_better = False
    order = 50

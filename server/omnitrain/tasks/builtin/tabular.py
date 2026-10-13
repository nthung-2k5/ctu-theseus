from omnitrain.tasks.base import AnnotationSpec, Task, TaskOutput
from omnitrain.tasks.builtin._common import CLASS, CLASSIFY, RECORD, SPLIT, PlannedTask, col


class TabularClassification(Task):
    id = "tabular_classification"
    label = "Tabular Classification"
    modality = "tabular"
    status = "stable"
    order = 30
    item_spec = RECORD
    annotation = CLASSIFY
    columns = [col(CLASS, "label"), col(SPLIT, "split")]
    output = TaskOutput(column=CLASS, kind="classification")


class TabularRegression(Task):
    id = "tabular_regression"
    label = "Tabular Regression"
    modality = "tabular"
    status = "stable"
    order = 40
    item_spec = RECORD
    annotation = AnnotationSpec(type="classification", requires_label_classes=False)
    columns = [col("target", "label"), col(SPLIT, "split")]
    output = TaskOutput(column="target", kind="regression")


class TabularClustering(PlannedTask):
    id = "tabular_clustering"
    label = "Tabular Clustering"
    modality = "tabular"
    order = 180
    annotation = AnnotationSpec(type="classification", requires_label_classes=False)


class TabularAnomalyDetection(PlannedTask):
    id = "tabular_anomaly_detection"
    label = "Tabular Anomaly Detection"
    modality = "tabular"
    order = 190
    annotation = AnnotationSpec(type="classification", requires_label_classes=False)

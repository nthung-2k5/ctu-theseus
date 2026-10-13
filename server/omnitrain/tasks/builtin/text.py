from omnitrain.tasks.base import AnnotationSpec, Task, TaskOutput
from omnitrain.tasks.builtin._common import INLINE, SPLIT, ClassificationTask, LlmTextTask, PlannedTask, col


class TextClassification(ClassificationTask):
    id = "text_classification"
    label = "Text Classification"
    modality = "text"
    status = "stable"
    order = 20
    item_spec = INLINE
    source_column = col("text", "inline_text")
    input_fields = ["text"]


class TokenClassification(Task):
    id = "token_classification"
    label = "Token Classification"
    modality = "text"
    order = 60
    item_spec = INLINE
    annotation = AnnotationSpec(type="token_tags", requires_label_classes=True)
    columns = [col("text", "inline_text"), col("tags", "label"), col(SPLIT, "split")]
    output = TaskOutput(column="tags", kind="tokens")
    input_fields = ["text"]


class TextGeneration(LlmTextTask):
    id = "text_generation"
    label = "Text Generation"
    order = 70
    input_fields = ["prompt"]
    output_field = "completion"


class Summarization(LlmTextTask):
    id = "summarization"
    label = "Summarization"
    order = 80
    input_fields = ["document"]
    output_field = "summary"


class SequenceToSequence(LlmTextTask):
    id = "sequence_to_sequence"
    label = "Sequence to Sequence"
    order = 90
    input_fields = ["source"]
    output_field = "target"


class QuestionAnswering(LlmTextTask):
    id = "question_answering"
    label = "Question Answering"
    order = 100
    input_fields = ["context", "question"]
    output_field = "answer"


class TextEmbedding(PlannedTask):
    id = "text_embedding"
    label = "Text Embedding"
    modality = "text"
    order = 200
    annotation = AnnotationSpec(type="classification", requires_label_classes=False)

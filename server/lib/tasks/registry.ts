import CONSTANTS from '@schema/constants.json'
import type { ProjectTask } from '@server/lib/enums'
import type { EncoderChoice, TaskDescriptor, TrainerKnobSpec } from './types'

const IMAGE_PATH_COLUMN = CONSTANTS.DATASET.IMAGE_PATH_COLUMN_NAME
const CLASS_COLUMN = CONSTANTS.DATASET.CLASS_COLUMN_NAME
const SPLIT_COLUMN = CONSTANTS.SPLIT_COLUMN_NAME

const defaultTrainerKnobs = (): TrainerKnobSpec => ({
  epochs: { default: 20, min: 1, max: 500 },
  batchSize: { default: 'auto', options: [16, 32, 64, 128, 256, 'auto'] },
  learningRate: { default: 0.001, min: 0.00001, max: 1 },
  earlyStopPatience: { default: 5, min: -1 },
})

/* ------------------------------------------------------------------ */
/*  Vision encoders — torchvision-backed Ludwig image encoders        */
/* ------------------------------------------------------------------ */
const visionEncoders: EncoderChoice[] = [
  { id: 'resnet18', label: 'ResNet-18', encoderType: 'resnet', pretrained: true, params: { model_variant: 18 } },
  { id: 'resnet50', label: 'ResNet-50', encoderType: 'resnet', pretrained: true, params: { model_variant: 50 } },
  {
    id: 'vit_base',
    label: 'ViT-Base/16',
    encoderType: 'vit',
    pretrained: true,
    params: { model_variant: 'base_patch16_224' },
  },
  {
    id: 'convnext_tiny',
    label: 'ConvNeXt-Tiny',
    encoderType: 'convnext',
    pretrained: true,
    params: { model_variant: 'tiny' },
  },
  {
    id: 'efficientnet_b0',
    label: 'EfficientNet-B0',
    encoderType: 'efficientnet',
    pretrained: true,
    params: { model_variant: 'b0' },
  },
  {
    id: 'mobilenet_v3_small',
    label: 'MobileNetV3-Small',
    encoderType: 'mobilenetv3',
    pretrained: true,
    params: { model_variant: 'small' },
  },
]

/* ------------------------------------------------------------------ */
/*  Text encoders — built-in Ludwig text encoders                     */
/* ------------------------------------------------------------------ */
const textEncoders: EncoderChoice[] = [
  { id: 'bert', label: 'BERT', encoderType: 'bert', pretrained: true },
  { id: 'distilbert', label: 'DistilBERT', encoderType: 'distilbert', pretrained: true },
  { id: 'roberta', label: 'RoBERTa', encoderType: 'roberta', pretrained: true },
  { id: 'stacked_cnn', label: 'Stacked CNN (train from scratch)', encoderType: 'stacked_cnn', pretrained: false },
]

/* ------------------------------------------------------------------ */
/*  Audio encoders — built-in Ludwig audio encoders                   */
/* ------------------------------------------------------------------ */
const audioEncoders: EncoderChoice[] = [
  { id: 'stacked_cnn', label: 'Stacked CNN', encoderType: 'stacked_cnn', pretrained: false },
  { id: 'rnn', label: 'RNN', encoderType: 'rnn', pretrained: false },
  { id: 'cnnrnn', label: 'CNN + RNN', encoderType: 'cnnrnn', pretrained: false },
]

/* ------------------------------------------------------------------ */
/*  Tier 1 — stable                                                   */
/* ------------------------------------------------------------------ */

const imageClassification: TaskDescriptor = {
  id: 'image_classification',
  label: 'Image Classification',
  modality: 'vision',
  backend: 'ludwig',
  status: 'stable',
  itemSpec: { payload: 'file', accept: ['image/jpeg', 'image/png'] },
  annotation: { type: 'classification', requiresLabelClasses: true },
  columns: [
    { name: IMAGE_PATH_COLUMN, kind: 'storage_uri' },
    { name: CLASS_COLUMN, kind: 'label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: IMAGE_PATH_COLUMN, type: 'image', column: IMAGE_PATH_COLUMN }],
    outputFeatures: () => [{ name: CLASS_COLUMN, type: 'category', column: CLASS_COLUMN }],
    encoders: visionEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const textClassification: TaskDescriptor = {
  id: 'text_classification',
  label: 'Text Classification',
  modality: 'text',
  backend: 'ludwig',
  status: 'stable',
  itemSpec: { payload: 'inline_text' },
  annotation: { type: 'classification', requiresLabelClasses: true },
  columns: [
    { name: 'text', kind: 'inline_text' },
    { name: CLASS_COLUMN, kind: 'label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: 'text', type: 'text', column: 'text' }],
    outputFeatures: () => [{ name: CLASS_COLUMN, type: 'category', column: CLASS_COLUMN }],
    encoders: textEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const tabularClassification: TaskDescriptor = {
  id: 'tabular_classification',
  label: 'Tabular Classification',
  modality: 'tabular',
  backend: 'ludwig',
  status: 'stable',
  itemSpec: { payload: 'record' },
  annotation: { type: 'classification', requiresLabelClasses: true },
  columns: [
    { name: CLASS_COLUMN, kind: 'label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    // Tabular feature columns are dynamic (dataset-defined), so input_features
    // is finalized by the compiler from the record schema at snapshot time.
    modelType: 'ecd',
    inputFeatures: () => [],
    outputFeatures: () => [{ name: CLASS_COLUMN, type: 'category', column: CLASS_COLUMN }],
    encoders: [],
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const tabularRegression: TaskDescriptor = {
  id: 'tabular_regression',
  label: 'Tabular Regression',
  modality: 'tabular',
  backend: 'ludwig',
  status: 'stable',
  itemSpec: { payload: 'record' },
  annotation: { type: 'classification', requiresLabelClasses: false },
  columns: [
    { name: 'target', kind: 'label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [],
    outputFeatures: () => [{ name: 'target', type: 'number', column: 'target' }],
    encoders: [],
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const audioClassification: TaskDescriptor = {
  id: 'audio_classification',
  label: 'Audio Classification',
  modality: 'audio',
  backend: 'ludwig',
  status: 'stable',
  itemSpec: { payload: 'file', accept: ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg'] },
  annotation: { type: 'classification', requiresLabelClasses: true },
  columns: [
    { name: 'audio_path', kind: 'storage_uri' },
    { name: CLASS_COLUMN, kind: 'label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: 'audio_path', type: 'audio', column: 'audio_path' }],
    outputFeatures: () => [{ name: CLASS_COLUMN, type: 'category', column: CLASS_COLUMN }],
    encoders: audioEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

/* ------------------------------------------------------------------ */
/*  Tier 2 — experimental (Ludwig `llm` model type, no bespoke UI)    */
/* ------------------------------------------------------------------ */

const llmTrainerKnobs = (): TrainerKnobSpec => ({
  epochs: { default: 3, min: 1, max: 50 },
  batchSize: { default: 1, options: [1, 2, 4, 8, 'auto'] },
  learningRate: { default: 0.0001, min: 0.000001, max: 0.01 },
  earlyStopPatience: { default: 3, min: -1 },
})

const tokenClassification: TaskDescriptor = {
  id: 'token_classification',
  label: 'Token Classification',
  modality: 'text',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'inline_text' },
  annotation: { type: 'token_tags', requiresLabelClasses: true },
  columns: [
    { name: 'text', kind: 'inline_text' },
    { name: 'tags', kind: 'label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: 'text', type: 'sequence', column: 'text' }],
    outputFeatures: () => [{ name: 'tags', type: 'sequence', column: 'tags' }],
    encoders: textEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const textGeneration: TaskDescriptor = {
  id: 'text_generation',
  label: 'Text Generation',
  modality: 'text',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'inline_text' },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: 'prompt', kind: 'inline_text' },
    { name: 'completion', kind: 'inline_text' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'llm',
    inputFeatures: () => [{ name: 'prompt', type: 'text', column: 'prompt' }],
    outputFeatures: () => [{ name: 'completion', type: 'text', column: 'completion' }],
    encoders: [],
    trainerKnobs: llmTrainerKnobs(),
  },
}

const summarization: TaskDescriptor = {
  id: 'summarization',
  label: 'Summarization',
  modality: 'text',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'inline_text' },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: 'document', kind: 'inline_text' },
    { name: 'summary', kind: 'inline_text' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'llm',
    inputFeatures: () => [{ name: 'document', type: 'text', column: 'document' }],
    outputFeatures: () => [{ name: 'summary', type: 'text', column: 'summary' }],
    encoders: [],
    trainerKnobs: llmTrainerKnobs(),
  },
}

const sequenceToSequence: TaskDescriptor = {
  id: 'sequence_to_sequence',
  label: 'Sequence to Sequence',
  modality: 'text',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'inline_text' },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: 'source', kind: 'inline_text' },
    { name: 'target', kind: 'inline_text' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'llm',
    inputFeatures: () => [{ name: 'source', type: 'text', column: 'source' }],
    outputFeatures: () => [{ name: 'target', type: 'text', column: 'target' }],
    encoders: [],
    trainerKnobs: llmTrainerKnobs(),
  },
}

const questionAnswering: TaskDescriptor = {
  id: 'question_answering',
  label: 'Question Answering',
  modality: 'text',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'inline_text' },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: 'context', kind: 'inline_text' },
    { name: 'question', kind: 'inline_text' },
    { name: 'answer', kind: 'inline_text' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'llm',
    inputFeatures: () => [
      { name: 'context', type: 'text', column: 'context' },
      { name: 'question', type: 'text', column: 'question' },
    ],
    outputFeatures: () => [{ name: 'answer', type: 'text', column: 'answer' }],
    encoders: [],
    trainerKnobs: llmTrainerKnobs(),
  },
}

/* ------------------------------------------------------------------ */
/*  Tier 3 — experimental (file input + free-text output, Ludwig ECD) */
/*                                                                     */
/*  Unlike object detection/segmentation/audio segmentation below,     */
/*  these have a real Ludwig backend: `TextOutputFeature` is a         */
/*  `SequenceOutputFeature` regardless of what feeds the combiner, so  */
/*  an image/audio encoder driving a `text`-type output is the same    */
/*  generic sequence-generation mechanism the Tier-2 `llm` text tasks  */
/*  use — just with a non-text encoder. `experimental` because this    */
/*  exact input/output combination is far less battle-tested than     */
/*  Ludwig's tabular-first usage.                                      */
/* ------------------------------------------------------------------ */

const imageCaptioning: TaskDescriptor = {
  id: 'image_captioning',
  label: 'Image Captioning',
  modality: 'vision',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'file', accept: ['image/jpeg', 'image/png'] },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: IMAGE_PATH_COLUMN, kind: 'storage_uri' },
    { name: 'caption', kind: 'text_sequence_label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: IMAGE_PATH_COLUMN, type: 'image', column: IMAGE_PATH_COLUMN }],
    outputFeatures: () => [{ name: 'caption', type: 'text', column: 'caption' }],
    encoders: visionEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const audioCaptioning: TaskDescriptor = {
  id: 'audio_captioning',
  label: 'Audio Captioning',
  modality: 'audio',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'file', accept: ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg'] },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: 'audio_path', kind: 'storage_uri' },
    { name: 'caption', kind: 'text_sequence_label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: 'audio_path', type: 'audio', column: 'audio_path' }],
    outputFeatures: () => [{ name: 'caption', type: 'text', column: 'caption' }],
    encoders: audioEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

const automaticSpeechRecognition: TaskDescriptor = {
  id: 'automatic_speech_recognition',
  label: 'Automatic Speech Recognition',
  modality: 'audio',
  backend: 'ludwig',
  status: 'experimental',
  itemSpec: { payload: 'file', accept: ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg'] },
  annotation: { type: 'text_sequence', requiresLabelClasses: false },
  columns: [
    { name: 'audio_path', kind: 'storage_uri' },
    { name: 'transcript', kind: 'text_sequence_label' },
    { name: SPLIT_COLUMN, kind: 'split' },
  ],
  ludwig: {
    modelType: 'ecd',
    inputFeatures: () => [{ name: 'audio_path', type: 'audio', column: 'audio_path' }],
    outputFeatures: () => [{ name: 'transcript', type: 'text', column: 'transcript' }],
    encoders: audioEncoders,
    trainerKnobs: defaultTrainerKnobs(),
  },
}

/* ------------------------------------------------------------------ */
/*  Tier 3 — planned, no Ludwig backend at all                        */
/*                                                                     */
/*  Genuinely blocked, not just unbuilt: Ludwig has no bounding-box or */
/*  segmentation-mask output feature type, so no amount of labeling   */
/*  UI makes these trainable through this platform's Ludwig-only      */
/*  backend — that would need an entirely different training path.    */
/* ------------------------------------------------------------------ */

function planned(id: ProjectTask, label: string, modality: TaskDescriptor['modality']): TaskDescriptor {
  return {
    id,
    label,
    modality,
    backend: 'unsupported',
    status: 'planned',
    itemSpec: { payload: 'file' },
    annotation: { type: 'classification', requiresLabelClasses: false },
    columns: [],
  }
}

const unsupportedTasks: TaskDescriptor[] = [
  {
    ...planned('object_detection', 'Object Detection', 'vision'),
    annotation: { type: 'bounding_box', requiresLabelClasses: true },
  },
  {
    ...planned('image_segmentation', 'Image Segmentation', 'vision'),
    annotation: { type: 'segmentation_mask', requiresLabelClasses: true },
  },
  {
    ...planned('audio_segmentation', 'Audio Segmentation', 'audio'),
    annotation: { type: 'segmentation_mask', requiresLabelClasses: true },
  },
  {
    ...planned('tabular_clustering', 'Tabular Clustering', 'tabular'),
    annotation: { type: 'classification', requiresLabelClasses: false },
  },
  {
    ...planned('tabular_anomaly_detection', 'Tabular Anomaly Detection', 'tabular'),
    annotation: { type: 'classification', requiresLabelClasses: false },
  },
  {
    ...planned('text_embedding', 'Text Embedding', 'text'),
    annotation: { type: 'classification', requiresLabelClasses: false },
  },
]

/* ------------------------------------------------------------------ */
/*  The registry                                                      */
/* ------------------------------------------------------------------ */

export const taskRegistry: Record<ProjectTask, TaskDescriptor> = Object.fromEntries(
  [
    imageClassification,
    textClassification,
    tabularClassification,
    tabularRegression,
    audioClassification,
    tokenClassification,
    textGeneration,
    summarization,
    sequenceToSequence,
    questionAnswering,
    imageCaptioning,
    audioCaptioning,
    automaticSpeechRecognition,
    ...unsupportedTasks,
  ].map((descriptor) => [descriptor.id, descriptor]),
) as Record<ProjectTask, TaskDescriptor>

export const ProjectTasks = [
  // Text modality
  'text_classification',
  'token_classification',
  'text_generation',
  'question_answering',
  'summarization',
  'sequence_to_sequence', // a.k.a. translation
  'text_embedding',

  // Vision modality
  'image_classification',
  'object_detection',
  'image_segmentation',
  'image_captioning',

  // Audio modality
  'audio_classification',
  'automatic_speech_recognition',
  'audio_segmentation',
  'audio_captioning',

  // Tabular modality
  'tabular_regression',
  'tabular_classification',
  'tabular_clustering',
  'tabular_anomaly_detection',
] as const

export type ProjectTask = (typeof ProjectTasks)[number]

export const DatasetModalities = ['text', 'vision', 'audio', 'tabular'] as const

export type DatasetModality = (typeof DatasetModalities)[number]

export const SplitTypes = ['train', 'test', 'validation'] as const

export type SplitType = (typeof SplitTypes)[number]

export const ImageFormats = ['jpeg', 'png'] as const

export type ImageFormat = (typeof ImageFormats)[number]

export const AudioCodecs = ['wav', 'mp3', 'flac', 'ogg'] as const

export type AudioCodec = (typeof AudioCodecs)[number]

export const AnnotationTypes = [
  'classification',
  'bounding_box',
  'segmentation_mask',
  'text_sequence',
  'token_tags',
  'preference_rank',
] as const

export type AnnotationType = (typeof AnnotationTypes)[number]

export const TrainingStatuses = ['queued', 'training', 'completed'] as const

export type TrainingStatus = (typeof TrainingStatuses)[number]

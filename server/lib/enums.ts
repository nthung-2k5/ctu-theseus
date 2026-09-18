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

export const TrainingStatuses = ['queued', 'running', 'succeeded', 'failed', 'canceled'] as const

export type TrainingStatus = (typeof TrainingStatuses)[number]

export const DatasetVersionStatuses = ['draft', 'building', 'ready', 'failed'] as const

export type DatasetVersionStatus = (typeof DatasetVersionStatuses)[number]

export const ExportFormats = ['onnx', 'torchscript'] as const

export type ExportFormat = (typeof ExportFormats)[number]

/**
 * `model`   — the raw artifact (onnx/torchscript) + preprocessing metadata.
 * `devkit`  — model tier + a generated inference client (source only — no
 *             project/build files; dependencies are documented in the
 *             bundle's README for you to add to your own project).
 * `app`     — a runnable client application: a Progressive Web App or a
 *             Flutter mobile app, both doing inference on-device/in-browser.
 *             Never a server.
 * devkit/app are ONNX-only — see server/lib/export/bundle.ts.
 */
export const ExportTiers = ['model', 'devkit', 'app'] as const

export type ExportTier = (typeof ExportTiers)[number]

/** `devkit` tier client languages — see server/lib/export/templates/{python,typescript,csharp,java}. */
export const DevkitLangs = ['python', 'typescript', 'csharp', 'java'] as const

export type DevkitLang = (typeof DevkitLangs)[number]

/** `app` tier targets — see server/lib/export/templates/{pwa,flutter}. */
export const AppTargets = ['pwa', 'flutter'] as const

export type AppTarget = (typeof AppTargets)[number]

export const ExportLangs = [...DevkitLangs, ...AppTargets] as const

export type ExportLang = (typeof ExportLangs)[number]

export const ExportStatuses = ['pending', 'converting', 'assembling', 'ready', 'failed'] as const

export type ExportStatus = (typeof ExportStatuses)[number]

/** Which split an evaluation report was computed on — `full` when the run's snapshot had no test/validation rows to fall back to. */
export const EvaluationSplits = [...SplitTypes, 'full'] as const

export type EvaluationSplit = (typeof EvaluationSplits)[number]

export const EvaluationStatuses = ['success', 'failed'] as const

export type EvaluationStatus = (typeof EvaluationStatuses)[number]

/**
 * `grid`   — every combination of the search space's candidate values,
 *            capped at `maxTrials` (see server/lib/sweep.ts's `expandGrid`).
 * `random` — `maxTrials` combinations sampled independently per knob.
 */
export const SweepStrategies = ['grid', 'random'] as const

export type SweepStrategy = (typeof SweepStrategies)[number]

/**
 * `running`   — at least one trial run hasn't reached a terminal status yet.
 * `completed` — every trial run finished (succeeded/failed/canceled) on its own.
 * `canceled`  — the sweep itself was canceled (see POST /sweeps/:sweepId/cancel);
 *               distinct from `completed` even if every trial happens to have
 *               already finished by the time cancel is requested.
 */
export const SweepStatuses = ['running', 'completed', 'canceled'] as const

export type SweepStatus = (typeof SweepStatuses)[number]

/**
 * `pending`   — dispatched, no terminal result persisted yet.
 * `success`   — worker published a result (see THESEUS_INFERENCE_RESULTS);
 *               `output` carries it.
 * `failed`    — worker published a failure, or delivery attempts on
 *               THESEUS_TASKS were exhausted; `error` carries the message.
 */
export const InferenceJobStatuses = ['pending', 'success', 'failed'] as const

export type InferenceJobStatus = (typeof InferenceJobStatuses)[number]

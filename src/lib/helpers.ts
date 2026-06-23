import type { DatasetModality, ProjectTask } from './enums'

/* ── Helper: derive modality from project task ── */
export function taskToModality(task: ProjectTask): DatasetModality {
  switch (task) {
    case 'text_classification':
    case 'token_classification':
    case 'text_generation':
    case 'question_answering':
    case 'summarization':
    case 'sequence_to_sequence':
    case 'text_embedding':
      return 'text'
    case 'image_classification':
    case 'object_detection':
    case 'image_segmentation':
    case 'image_captioning':
      return 'vision'
    case 'audio_classification':
    case 'automatic_speech_recognition':
    case 'audio_segmentation':
    case 'audio_captioning':
      return 'audio'
    case 'tabular_regression':
    case 'tabular_classification':
    case 'tabular_clustering':
    case 'tabular_anomaly_detection':
      return 'tabular'
    default:
      throw new Error(`Unknown task: ${task}`)
  }
}

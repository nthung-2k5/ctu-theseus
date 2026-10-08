import {
  ArrowsLeftRightIcon,
  ArticleIcon,
  BoundingBoxIcon,
  ChartLineIcon,
  ChartScatterIcon,
  ImageSquareIcon,
  ImagesIcon,
  ListChecksIcon,
  MicrophoneIcon,
  QuestionIcon,
  ScissorsIcon,
  SelectionIcon,
  SubtitlesIcon,
  TableIcon,
  TagIcon,
  TextAaIcon,
  TreeStructureIcon,
  VectorThreeIcon,
  WarningIcon,
  WaveformIcon,
} from '@phosphor-icons/react'
import type { ProjectTask } from '@public/lib/api/enums'

export interface TaskMeta {
  icon: typeof TagIcon
  description: string
}

/**
 * Presentation-only copy for the task picker: an icon and a one-line description per task. The data
 * contract of a task lives in the task registry; this is only what a person reads when choosing one.
 */
export const TASK_META: Record<ProjectTask, TaskMeta> = {
  image_classification: {
    icon: ImageSquareIcon,
    description: 'Assign each image to one category, such as defective or fine.',
  },
  object_detection: {
    icon: BoundingBoxIcon,
    description: 'Find objects in an image and draw a labelled box around each one.',
  },
  image_segmentation: {
    icon: SelectionIcon,
    description: 'Label every pixel of an image to outline the exact shape of objects.',
  },
  image_captioning: {
    icon: ImagesIcon,
    description: 'Describe what an image shows in a sentence.',
  },
  text_classification: {
    icon: TagIcon,
    description: 'Assign a piece of text to a category, such as sentiment or topic.',
  },
  token_classification: {
    icon: TextAaIcon,
    description: 'Tag individual words, for example names, places or dates.',
  },
  text_generation: {
    icon: ArticleIcon,
    description: 'Continue or write text from a prompt.',
  },
  summarization: {
    icon: ListChecksIcon,
    description: 'Condense a long text into a short summary.',
  },
  sequence_to_sequence: {
    icon: ArrowsLeftRightIcon,
    description: 'Turn one piece of text into another, such as translation or rewriting.',
  },
  question_answering: {
    icon: QuestionIcon,
    description: 'Answer a question using the text you give the model.',
  },
  text_embedding: {
    icon: VectorThreeIcon,
    description: 'Turn text into vectors for search and similarity.',
  },
  tabular_classification: {
    icon: TableIcon,
    description: 'Predict a category from the columns of a table row.',
  },
  tabular_regression: {
    icon: ChartLineIcon,
    description: 'Predict a number from the columns of a table row.',
  },
  tabular_clustering: {
    icon: TreeStructureIcon,
    description: 'Group similar rows together without any labels.',
  },
  tabular_anomaly_detection: {
    icon: WarningIcon,
    description: 'Flag rows that look unlike the rest of the data.',
  },
  audio_classification: {
    icon: WaveformIcon,
    description: 'Assign an audio clip to a category, such as a sound or a speaker.',
  },
  audio_captioning: {
    icon: SubtitlesIcon,
    description: 'Describe the sounds in a clip in a sentence.',
  },
  automatic_speech_recognition: {
    icon: MicrophoneIcon,
    description: 'Transcribe spoken audio into text.',
  },
  audio_segmentation: {
    icon: ScissorsIcon,
    description: 'Mark the time ranges in a recording where each sound occurs.',
  },
}

export const FALLBACK_TASK_META: TaskMeta = { icon: ChartScatterIcon, description: '' }

export function getTaskMeta(task: string): TaskMeta {
  return TASK_META[task as ProjectTask] ?? FALLBACK_TASK_META
}

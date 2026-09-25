import { ChatTextIcon, ImageSquareIcon, TableIcon, WaveformIcon } from '@phosphor-icons/react'
import type { DatasetModality } from '@public/lib/api/enums'

export interface ModalityMeta {
  label: string
  color: string
  icon: typeof ImageSquareIcon
  description: string
}

/** Per-modality label, colour and icon. One copy, shared by the projects list, overview and pickers. */
export const MODALITY_META: Record<DatasetModality, ModalityMeta> = {
  vision: {
    label: 'Vision',
    color: 'green',
    icon: ImageSquareIcon,
    description: 'Image classification, detection & visual understanding',
  },
  text: {
    label: 'Text',
    color: 'blue',
    icon: ChatTextIcon,
    description: 'NLP, sequence tagging, classification & generation',
  },
  audio: {
    label: 'Audio',
    color: 'orange',
    icon: WaveformIcon,
    description: 'Sound classification & acoustic event labeling',
  },
  tabular: {
    label: 'Tabular',
    color: 'grape',
    icon: TableIcon,
    description: 'Structured records, numerical & category prediction',
  },
}

import { PARTITION_COLORS } from '@public/lib/palette'

/** Shared UI color maps used across the dataset/snapshot/training views. */

/**
 * The one split palette. `components/training/constants.ts` used to carry a
 * second, differently-coloured copy, so the same split read green in the item
 * grid and blue in the metrics chart.
 */
export const SPLIT_COLORS: Record<string, string> = {
  train: PARTITION_COLORS.train,
  validation: PARTITION_COLORS.validation,
  test: PARTITION_COLORS.test,
}

/** Chart-shade variants of the same palette, for series lines/areas. */
export const SPLIT_CHART_COLORS: Record<string, string> = {
  train: PARTITION_COLORS.train,
  validation: PARTITION_COLORS.validation,
  test: PARTITION_COLORS.test,
}

/**
 * Split picker options, in the order they're offered. Previously duplicated in
 * UploadPage and TabularCsvImporter, which had already drifted in ordering.
 */
export const SPLIT_OPTIONS = [
  { value: 'train', label: 'Train' },
  { value: 'validation', label: 'Validation' },
  { value: 'test', label: 'Test' },
]

/**
 * Status colors for training run statuses.
 */
export const STATUS_COLORS: Record<string, string> = {
  queued: 'gray',
  running: 'cyan',
  succeeded: 'teal',
  failed: 'red',
  canceled: 'gray',
}

// Split colors live in lib/constants.ts. This file used to define a second,
// differently-coloured map, so the same split showed green in the item grid
// and blue in the metric chart.
export { SPLIT_CHART_COLORS as SPLIT_COLORS } from '@public/lib/constants'

/**
 * Human-readable labels for the metric keys the trainer backends report. Metrics not
 * listed here (e.g. per-output-feature custom names) fall back to a
 * title-cased rendering of the raw key — see `formatMetricLabel`.
 */
export const METRIC_LABELS: Record<string, string> = {
  loss: 'Loss',
  accuracy: 'Accuracy',
  accuracy_micro: 'Accuracy (micro)',
  precision: 'Precision',
  recall: 'Recall',
  specificity: 'Specificity',
  roc_auc: 'ROC AUC',
  hits_at_k: 'Hits @ K',
  root_mean_squared_error: 'RMSE',
  root_mean_squared_percentage_error: 'RMSPE',
  mean_squared_error: 'MSE',
  mean_absolute_error: 'MAE',
  mean_absolute_percentage_error: 'MAPE',
  mean_absolute_scaled_error: 'MASE',
  symmetric_mean_absolute_percentage_error: 'SMAPE',
  r2: 'R²',
  huber: 'Huber Loss',
  binary_weighted_cross_entropy: 'Binary Weighted Cross-Entropy',
  softmax_cross_entropy: 'Softmax Cross-Entropy',
  sequence_softmax_cross_entropy: 'Sequence Softmax Cross-Entropy',
  next_token_softmax_cross_entropy: 'Next-Token Softmax Cross-Entropy',
  sigmoid_cross_entropy: 'Sigmoid Cross-Entropy',
  token_accuracy: 'Token Accuracy',
  sequence_accuracy: 'Sequence Accuracy',
  perplexity: 'Perplexity',
  next_token_perplexity: 'Next-Token Perplexity',
  bleu: 'BLEU',
  rouge: 'ROUGE',
  word_error_rate: 'Word Error Rate',
  char_error_rate: 'Character Error Rate',
  jaccard: 'Jaccard Similarity',
  corn: 'CORN Loss',
  anomaly_auroc: 'Anomaly ROC AUC',
  f1_max: 'F1 (max)',
  // The YOLO backend's own metric names (see server/omnitrain/plugins/backends/yolo/train.py).
  top5Accuracy: 'Top-5 accuracy',
  box_loss: 'Box loss',
  cls_loss: 'Class loss',
  dfl_loss: 'Distribution focal loss',
  mAP50: 'mAP@50',
  'mAP50-95': 'mAP@50-95',
}

/** Falls back to a title-cased rendering of unknown metric keys. */
export function formatMetricLabel(metricName: string): string {
  if (metricName === 'loss') return 'Average Loss'

  const key = metricName.replace('class.', '')
  // Backends emit metrics beyond the table above (per-output-feature names,
  // `accuracy_macro`, `f1_micro`, `combined`, ...). Without this fallback they
  // rendered as an empty <Select> option and a blank metrics-table cell.
  return (
    METRIC_LABELS[key] ??
    key
      .split('_')
      .map((word) => (word ? word.charAt(0).toUpperCase() + word.slice(1) : word))
      .join(' ')
  )
}

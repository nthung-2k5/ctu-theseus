import { t } from 'elysia'

// export const TrainingSchema = t.Object({
//   runId: t.String({ format: 'uuid', description: 'The run ID' }),

//   inputFeatures: t.Array(
//     t.Object({
//       name: t.String({ description: 'The column name in the training dataset' }),
//       type: t.Union([
//         t.Literal('image'),
//         t.Literal('text'),
//         t.Literal('numerical'),
//         t.Literal('binary'),
//         t.Literal('category'),
//       ]),
//       encoder: t.String({ description: 'The chosen predefined model/architecture string' }),
//     }),
//     { minItems: 1 },
//   ),

//   outputFeatures: t.Array(
//     t.Object({
//       name: t.String(),
//       type: t.Union([t.Literal('category'), t.Literal('numerical'), t.Literal('binary')]),
//       loss: t.Optional(
//         t.Union(
//           [
//             t.Literal('softmax_cross_entropy', { title: 'Softmax Cross-Entropy' }),

//             // ── OOD loss functions ──
//             t.Literal('corn', { title: 'CORN' }),
//             t.Literal('entropic_open_set', { title: 'Entropic Open-Set' }),
//             t.Literal('objectosphere', { title: 'Objectosphere' }),

//             // ── Modern improvements to softmax-based losses ──
//             t.Literal('focal', { title: 'Focal Loss' }),
//             t.Literal('poly', { title: 'Polynomial Loss' }),

//             // ── Sparse Probabilities ──
//             t.Literal('entmax15', { title: 'Entmax15' }),
//           ],
//           { default: 'softmax_cross_entropy' },
//         ),
//       ),
//     }),
//     { minItems: 1 },
//   ),

//   trainer: t.Object({
//     epochs: t.Integer({ minimum: 1, default: 50 }),
//     // Ludwig allows explicit integer batch sizes or setting it to automated tuning
//     batchSize: t.Union([t.Integer({ minimum: 1 }), t.Literal('auto')], { default: 'auto' }),
//     optimizer: t.Object({
//       type: t.Union(
//         [
//           // Standard optimizers
//           t.Literal('sgd', { title: 'SGD' }),
//           t.Literal('adamw', { title: 'AdamW' }),

//           // Recommended optimizers
//           t.Literal('schedule_free_adamw', { title: 'Schedule-Free AdamW' }),

//           // 8-bit quantized versions
//           t.Literal('sgd_8bit', { title: 'SGD 8-bit Quantized' }),
//           t.Literal('adamw_8bit', { title: 'AdamW 8-bit Quantized' }),

//           // Paged optimizers (better memory efficiency)
//           t.Literal('paged_adamw', { title: 'Paged AdamW' }),
//           t.Literal('paged_adamw_8bit', { title: 'Paged AdamW 8-bit Quantized' }),
//         ],
//         {
//           default: 'schedule_free_adamw',
//         },
//       ),
//       learningRate: t.Number({ exclusiveMinimum: 0, default: 0.001 }),
//     }),

//     scheduler: t.Optional(
//       t.Object({
//         decay: t.Union(
//           [
//             t.Literal('linear', { title: 'Linear' }),
//             t.Literal('cosine', { title: 'Cosine' }),
//             t.Literal('one_cycle', { title: 'One Cycle' }),
//             // Niche, mostly for semantic segmentation
//             t.Literal('polynomial', { title: 'Polynomial' }),
//           ],
//           {
//             default: 'cosine',
//           },
//         ),
//         warmupEpochs: t.Optional(t.Integer({ minimum: 0, default: 0 })),
//       }),
//     ),

//     earlyStopping: t.Integer({
//       minimum: -1,
//       default: 5,
//       description: 'Number of epochs without improvement before halting. Use -1 to disable.',
//     }),
//   }),
// }, { title: 'LudwigTrainingConfiguration' })

export const TrainingSchema = t.Object(
  {
    runId: t.String({ format: 'uuid', description: 'The run ID' }),
    datasetVersionId: t.String({ format: 'uuid', description: 'The dataset version ID' }),
  },
  { title: 'LudwigTrainingConfiguration' },
)

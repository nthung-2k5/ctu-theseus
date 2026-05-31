/* ── Model Families ── */

export interface ModelFamily {
  id: string
  name: string
  variants: { id: string; label: string }[]
}

export const MODEL_FAMILIES: ModelFamily[] = [
  {
    id: 'resnet',
    name: 'ResNet',
    variants: [
      { id: 'resnet18', label: 'ResNet-18' },
      { id: 'resnet34', label: 'ResNet-34' },
      { id: 'resnet50', label: 'ResNet-50' },
      { id: 'resnet101', label: 'ResNet-101' },
      { id: 'resnet152', label: 'ResNet-152' },
    ],
  },
  {
    id: 'vit',
    name: 'Vision Transformer (ViT)',
    variants: [
      { id: 'vit_tiny', label: 'ViT-Tiny' },
      { id: 'vit_small', label: 'ViT-Small' },
      { id: 'vit_base', label: 'ViT-Base' },
      { id: 'vit_large', label: 'ViT-Large' },
    ],
  },
  {
    id: 'convnext',
    name: 'ConvNeXt',
    variants: [
      { id: 'convnext_tiny', label: 'ConvNeXt-Tiny' },
      { id: 'convnext_small', label: 'ConvNeXt-Small' },
      { id: 'convnext_base', label: 'ConvNeXt-Base' },
      { id: 'convnext_large', label: 'ConvNeXt-Large' },
    ],
  },
  {
    id: 'mobilenet',
    name: 'MobileNet',
    variants: [
      { id: 'mobilenet_v2', label: 'MobileNetV2' },
      { id: 'mobilenet_v3_small', label: 'MobileNetV3-Small' },
      { id: 'mobilenet_v3_large', label: 'MobileNetV3-Large' },
    ],
  },
  {
    id: 'efficientnet',
    name: 'EfficientNet',
    variants: [
      { id: 'efficientnet_b0', label: 'EfficientNet-B0' },
      { id: 'efficientnet_b1', label: 'EfficientNet-B1' },
      { id: 'efficientnet_b2', label: 'EfficientNet-B2' },
      { id: 'efficientnet_b3', label: 'EfficientNet-B3' },
      { id: 'efficientnet_b4', label: 'EfficientNet-B4' },
    ],
  },
]

/* ── Preprocessing & Augmentation definitions ── */

export interface ParamDef {
  key: string
  label: string
  type: 'number' | 'boolean'
  default: number | boolean
  min?: number
  max?: number
  step?: number
  suffix?: string
}

export interface OptionDef {
  id: string
  name: string
  params: ParamDef[]
}

export const PREPROCESSING_OPTIONS: OptionDef[] = [
  { id: 'autoOrient', name: 'Auto Orient', params: [] },
  {
    id: 'resize',
    name: 'Resize',
    params: [
      { key: 'width', label: 'Width', type: 'number', default: 640, min: 32, max: 1920, step: 32 },
      { key: 'height', label: 'Height', type: 'number', default: 640, min: 32, max: 1920, step: 32 },
    ],
  },
  { id: 'grayscale', name: 'Grayscale', params: [] },
]

export const AUGMENTATION_OPTIONS: OptionDef[] = [
  {
    id: 'flip',
    name: 'Flip',
    params: [
      { key: 'horizontal', label: 'Horizontal', type: 'boolean', default: true },
      { key: 'vertical', label: 'Vertical', type: 'boolean', default: false },
    ],
  },
  { id: 'rotate90', name: '90° Rotate', params: [] },
  {
    id: 'rotate',
    name: 'Rotate',
    params: [{ key: 'degrees', label: 'Max degrees', type: 'number', default: 15, min: 1, max: 45, suffix: '°' }],
  },
  {
    id: 'crop',
    name: 'Crop',
    params: [
      { key: 'minPercent', label: 'Min %', type: 'number', default: 0, min: 0, max: 99, suffix: '%' },
      { key: 'maxPercent', label: 'Max %', type: 'number', default: 20, min: 1, max: 50, suffix: '%' },
    ],
  },
  {
    id: 'rotation',
    name: 'Rotation',
    params: [{ key: 'degrees', label: 'Max degrees', type: 'number', default: 30, min: 1, max: 180, suffix: '°' }],
  },
  {
    id: 'shear',
    name: 'Shear',
    params: [{ key: 'degrees', label: 'Max degrees', type: 'number', default: 15, min: 1, max: 45, suffix: '°' }],
  },
  {
    id: 'augGrayscale',
    name: 'Grayscale',
    params: [{ key: 'probability', label: 'Probability', type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 }],
  },
  {
    id: 'hue',
    name: 'Hue',
    params: [{ key: 'range', label: 'Range', type: 'number', default: 20, min: 1, max: 180, suffix: '°' }],
  },
  {
    id: 'saturation',
    name: 'Saturation',
    params: [{ key: 'range', label: 'Range %', type: 'number', default: 25, min: 1, max: 100, suffix: '%' }],
  },
  {
    id: 'brightness',
    name: 'Brightness',
    params: [{ key: 'range', label: 'Range %', type: 'number', default: 20, min: 1, max: 100, suffix: '%' }],
  },
  {
    id: 'exposure',
    name: 'Exposure',
    params: [{ key: 'range', label: 'Range %', type: 'number', default: 15, min: 1, max: 100, suffix: '%' }],
  },
  {
    id: 'blur',
    name: 'Blur',
    params: [
      { key: 'maxRadius', label: 'Max radius', type: 'number', default: 2, min: 0.5, max: 10, step: 0.5, suffix: 'px' },
    ],
  },
  {
    id: 'noise',
    name: 'Noise',
    params: [{ key: 'maxPercent', label: 'Max %', type: 'number', default: 5, min: 1, max: 50, suffix: '%' }],
  },
  {
    id: 'cutout',
    name: 'Cutout',
    params: [
      { key: 'count', label: 'Count', type: 'number', default: 3, min: 1, max: 10 },
      { key: 'sizePercent', label: 'Size %', type: 'number', default: 10, min: 1, max: 50, suffix: '%' },
    ],
  },
  {
    id: 'motionBlur',
    name: 'Motion Blur',
    params: [{ key: 'maxKernel', label: 'Max kernel', type: 'number', default: 7, min: 3, max: 21, step: 2 }],
  },
  {
    id: 'cameraGain',
    name: 'Camera Gain',
    params: [{ key: 'range', label: 'Range', type: 'number', default: 15, min: 1, max: 50 }],
  },
]

/* ── Status colors ── */

export const STATUS_COLORS: Record<string, string> = {
  queued: 'yellow',
  training: 'blue',
  completed: 'green',
  failed: 'red',
}

/* ── Types ── */

export interface TrainingMetricPoint {
  epoch: number
  trainLoss: number
  valLoss: number
  accuracy: number
  mAP: number
}

export type OptionState = Record<string, { enabled: boolean; params: Record<string, number | boolean> }>

export interface TrainingRunData {
  id: string
  name: string
  modelId: string
  variantId: string
  epochs: number
  batchSize: number
  learningRate: number
  imageSize: number
  status: 'queued' | 'training' | 'completed' | 'failed'
  metrics: TrainingMetricPoint[]
  preprocessing: OptionState
  augmentation: OptionState
  createdAt: Date
  startedAt?: Date
  completedAt?: Date
  error?: string
}

export interface TrainingVersionConfig {
  modelId: string
  variantId: string
  epochs: number
  batchSize: number
  learningRate: number
  imageSize: number
  preprocessing: OptionState
  augmentation: OptionState
}

/* ── Helpers ── */

export function getVariantLabel(modelId: string, variantId: string): string {
  const family = MODEL_FAMILIES.find((f) => f.id === modelId)
  return family?.variants.find((v) => v.id === variantId)?.label ?? variantId
}

export function getFamilyName(modelId: string): string {
  return MODEL_FAMILIES.find((f) => f.id === modelId)?.name ?? modelId
}

export function buildDefaultOptionState(options: OptionDef[], enabledIds: string[] = []): OptionState {
  const state: OptionState = {}
  for (const opt of options) {
    const params: Record<string, number | boolean> = {}
    for (const p of opt.params) params[p.key] = p.default
    state[opt.id] = { enabled: enabledIds.includes(opt.id), params }
  }
  return state
}

/* ── Mock data ── */

function generateMetrics(epochs: number, quality: 'high' | 'medium'): TrainingMetricPoint[] {
  const k = quality === 'high' ? 0.065 : 0.045
  const maxAcc = quality === 'high' ? 0.945 : 0.885
  const maxMap = quality === 'high' ? 0.92 : 0.86
  const baseTL = quality === 'high' ? 0.12 : 0.18
  const baseVL = quality === 'high' ? 0.2 : 0.3

  return Array.from({ length: epochs }, (_, i) => {
    const t = i + 1
    return {
      epoch: t,
      trainLoss: +(2.5 * Math.exp(-k * t) + baseTL + (Math.random() - 0.5) * 0.04).toFixed(4),
      valLoss: +(2.8 * Math.exp(-(k - 0.01) * t) + baseVL + (Math.random() - 0.5) * 0.06).toFixed(4),
      accuracy: +Math.min(maxAcc, 0.25 + (maxAcc - 0.25) * (1 - Math.exp(-0.08 * t)) + (Math.random() - 0.5) * 0.015).toFixed(4),
      mAP: +Math.min(maxMap, 0.2 + (maxMap - 0.2) * (1 - Math.exp(-0.07 * t)) + (Math.random() - 0.5) * 0.02).toFixed(4),
    }
  })
}

export function createMockRuns(): TrainingRunData[] {
  const now = Date.now()
  return [
    {
      id: 'mock-run-1',
      name: 'ResNet-50 v1',
      modelId: 'resnet',
      variantId: 'resnet50',
      epochs: 50,
      batchSize: 32,
      learningRate: 0.001,
      imageSize: 224,
      status: 'completed',
      metrics: generateMetrics(50, 'high'),
      preprocessing: {
        autoOrient: { enabled: true, params: {} },
        resize: { enabled: true, params: { width: 224, height: 224 } },
      },
      augmentation: {
        flip: { enabled: true, params: { horizontal: true, vertical: false } },
        rotate: { enabled: true, params: { degrees: 15 } },
        brightness: { enabled: true, params: { range: 20 } },
      },
      createdAt: new Date(now - 2 * 86_400_000),
      startedAt: new Date(now - 2 * 86_400_000 + 5000),
      completedAt: new Date(now - 2 * 86_400_000 + 3_600_000),
    },
    {
      id: 'mock-run-2',
      name: 'EfficientNet-B2 v1',
      modelId: 'efficientnet',
      variantId: 'efficientnet_b2',
      epochs: 30,
      batchSize: 16,
      learningRate: 0.0005,
      imageSize: 260,
      status: 'completed',
      metrics: generateMetrics(30, 'medium'),
      preprocessing: {
        autoOrient: { enabled: true, params: {} },
        resize: { enabled: true, params: { width: 260, height: 260 } },
      },
      augmentation: {
        flip: { enabled: true, params: { horizontal: true, vertical: true } },
        noise: { enabled: true, params: { maxPercent: 3 } },
      },
      createdAt: new Date(now - 3 * 86_400_000),
      startedAt: new Date(now - 3 * 86_400_000 + 5000),
      completedAt: new Date(now - 3 * 86_400_000 + 2_400_000),
    },
    {
      id: 'mock-run-3',
      name: 'ViT-Small v1',
      modelId: 'vit',
      variantId: 'vit_small',
      epochs: 50,
      batchSize: 8,
      learningRate: 0.0001,
      imageSize: 384,
      status: 'failed',
      metrics: generateMetrics(12, 'medium'),
      preprocessing: {
        autoOrient: { enabled: true, params: {} },
        resize: { enabled: true, params: { width: 384, height: 384 } },
      },
      augmentation: {
        flip: { enabled: true, params: { horizontal: true, vertical: false } },
      },
      createdAt: new Date(now - 86_400_000),
      startedAt: new Date(now - 86_400_000 + 5000),
      error: 'CUDA out of memory. Tried to allocate 2.50 GiB.',
    },
  ]
}

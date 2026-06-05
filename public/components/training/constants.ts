/* ── Model Families ── */

import { MantineColor } from '@mantine/core';

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

// export interface OptionDef {
//   id: string
//   name: string
//   params: ParamDef[]
// }

// export const PREPROCESSING_OPTIONS: OptionDef[] = [
//   { id: 'autoOrient', name: 'Auto Orient', params: [] },
//   {
//     id: 'resize',
//     name: 'Resize',
//     params: [
//       { key: 'width', label: 'Width', type: 'number', default: 640, min: 32, max: 1920, step: 32 },
//       { key: 'height', label: 'Height', type: 'number', default: 640, min: 32, max: 1920, step: 32 },
//     ],
//   },
//   { id: 'grayscale', name: 'Grayscale', params: [] },
// ]

// export const AUGMENTATION_OPTIONS: OptionDef[] = [
//   {
//     id: 'flip',
//     name: 'Flip',
//     params: [
//       { key: 'horizontal', label: 'Horizontal', type: 'boolean', default: true },
//       { key: 'vertical', label: 'Vertical', type: 'boolean', default: false },
//     ],
//   },
//   { id: 'rotate90', name: '90° Rotate', params: [] },
//   {
//     id: 'rotate',
//     name: 'Rotate',
//     params: [{ key: 'degrees', label: 'Max degrees', type: 'number', default: 15, min: 1, max: 45, suffix: '°' }],
//   },
//   {
//     id: 'crop',
//     name: 'Crop',
//     params: [
//       { key: 'minPercent', label: 'Min %', type: 'number', default: 0, min: 0, max: 99, suffix: '%' },
//       { key: 'maxPercent', label: 'Max %', type: 'number', default: 20, min: 1, max: 50, suffix: '%' },
//     ],
//   },
//   {
//     id: 'rotation',
//     name: 'Rotation',
//     params: [{ key: 'degrees', label: 'Max degrees', type: 'number', default: 30, min: 1, max: 180, suffix: '°' }],
//   },
//   {
//     id: 'shear',
//     name: 'Shear',
//     params: [{ key: 'degrees', label: 'Max degrees', type: 'number', default: 15, min: 1, max: 45, suffix: '°' }],
//   },
//   {
//     id: 'augGrayscale',
//     name: 'Grayscale',
//     params: [{ key: 'probability', label: 'Probability', type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 }],
//   },
//   {
//     id: 'hue',
//     name: 'Hue',
//     params: [{ key: 'range', label: 'Range', type: 'number', default: 20, min: 1, max: 180, suffix: '°' }],
//   },
//   {
//     id: 'saturation',
//     name: 'Saturation',
//     params: [{ key: 'range', label: 'Range %', type: 'number', default: 25, min: 1, max: 100, suffix: '%' }],
//   },
//   {
//     id: 'brightness',
//     name: 'Brightness',
//     params: [{ key: 'range', label: 'Range %', type: 'number', default: 20, min: 1, max: 100, suffix: '%' }],
//   },
//   {
//     id: 'exposure',
//     name: 'Exposure',
//     params: [{ key: 'range', label: 'Range %', type: 'number', default: 15, min: 1, max: 100, suffix: '%' }],
//   },
//   {
//     id: 'blur',
//     name: 'Blur',
//     params: [
//       { key: 'maxRadius', label: 'Max radius', type: 'number', default: 2, min: 0.5, max: 10, step: 0.5, suffix: 'px' },
//     ],
//   },
//   {
//     id: 'noise',
//     name: 'Noise',
//     params: [{ key: 'maxPercent', label: 'Max %', type: 'number', default: 5, min: 1, max: 50, suffix: '%' }],
//   },
//   {
//     id: 'cutout',
//     name: 'Cutout',
//     params: [
//       { key: 'count', label: 'Count', type: 'number', default: 3, min: 1, max: 10 },
//       { key: 'sizePercent', label: 'Size %', type: 'number', default: 10, min: 1, max: 50, suffix: '%' },
//     ],
//   },
//   {
//     id: 'motionBlur',
//     name: 'Motion Blur',
//     params: [{ key: 'maxKernel', label: 'Max kernel', type: 'number', default: 7, min: 3, max: 21, step: 2 }],
//   },
//   {
//     id: 'cameraGain',
//     name: 'Camera Gain',
//     params: [{ key: 'range', label: 'Range', type: 'number', default: 15, min: 1, max: 50 }],
//   },
// ]

/* ── Status colors ── */

export const STATUS_COLORS: Record<"preparing" | "queued" | "training" | "completed" | "failed" | "stopped", MantineColor> = {
  preparing: 'gray',
  queued: 'yellow',
  training: 'blue',
  completed: 'green',
  failed: 'red.9',
  stopped: 'red',
}

/* ── Types ── */

export interface TrainingMetricPoint {
  epoch: number
  trainingLoss: number
  validationLoss: number
  accuracy: number
  mAP: number
}

// export type OptionState = Record<string, { enabled: boolean; params: Record<string, number | boolean> }>

/* ── Helpers ── */

export function getVariantLabel(modelId: string, variantId: string): string {
  const family = MODEL_FAMILIES.find((f) => f.id === modelId)
  return family?.variants.find((v) => v.id === variantId)?.label ?? variantId
}

export function getFamilyName(modelId: string): string {
  return MODEL_FAMILIES.find((f) => f.id === modelId)?.name ?? modelId
}

// export function buildDefaultOptionState(options: OptionDef[], enabledIds: string[] = []): OptionState {
//   const state: OptionState = {}
//   for (const opt of options) {
//     const params: Record<string, number | boolean> = {}
//     for (const p of opt.params) params[p.key] = p.default
//     state[opt.id] = { enabled: enabledIds.includes(opt.id), params }
//   }
//   return state
// }


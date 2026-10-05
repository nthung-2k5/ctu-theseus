import type { AnnotationType } from '@public/lib/api/enums'
import type { Annotation } from '@public/store/types'

/** The task registry's `annotation` field (see `TaskDescriptor` in lib/tasks.ts). */
export interface GroundTruthSpec {
  type: AnnotationType
  requiresLabelClasses: boolean
}

/** What the UI shows and edits for one task's ground truth. Derived from the registry, never from a task id. */
export type GroundTruthKind = 'class' | 'number' | 'text' | 'tags' | 'other'

/** Matches what the classification tasks did before the registry-driven drawer, so callers that pass no spec keep working. */
export const CLASS_GROUND_TRUTH: GroundTruthSpec = { type: 'classification', requiresLabelClasses: true }

export const groundTruthKind = (spec: GroundTruthSpec): GroundTruthKind => {
  switch (spec.type) {
    // A regression target is a classification-typed annotation with no class; its number is `labelStructured.value`.
    case 'classification':
      return spec.requiresLabelClasses ? 'class' : 'number'
    case 'text_sequence':
      return 'text'
    case 'token_tags':
      return 'tags'
    default:
      return 'other'
  }
}

export const findGroundTruth = (annotations: Annotation[] | undefined, spec: GroundTruthSpec) =>
  annotations?.find((a) => a.annotationType === spec.type)

export const targetValue = (annotation: Annotation | undefined): number | null => {
  const value = (annotation?.labelStructured as { value?: unknown } | null | undefined)?.value
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** A one-line, human-readable rendering of an item's ground truth, or null when it has none yet. */
export const groundTruthLabel = (
  annotations: Annotation[] | undefined,
  spec: GroundTruthSpec,
  classNameById: Map<string, string>,
): string | null => {
  const annotation = findGroundTruth(annotations, spec)
  if (!annotation) return null
  switch (groundTruthKind(spec)) {
    case 'class':
      return classNameById.get(annotation.classId ?? '') ?? null
    case 'number': {
      const value = targetValue(annotation)
      return value === null ? null : String(value)
    }
    case 'text':
      return annotation.labelTextSequence || null
    case 'tags': {
      const tags = annotation.labelStructured
      return Array.isArray(tags) ? `${tags.length} tag${tags.length === 1 ? '' : 's'}` : null
    }
    default:
      return null
  }
}

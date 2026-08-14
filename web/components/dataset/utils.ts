import type { Annotation } from '@public/store/types'

/** Every stable task's ground truth is one `classification`-type annotation per item (see server/lib/tasks/registry.ts). */
export function findClassificationAnnotation(annotations: Annotation[] | undefined): Annotation | undefined {
  return annotations?.find((a) => a.annotationType === 'classification')
}

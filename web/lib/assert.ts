// biome-ignore lint/suspicious/noExplicitAny: <explanation>
export function assert(condition: any, msg?: string): asserts condition {
  if (!condition) {
    throw new Error(msg || 'Assertion failed')
  }
}

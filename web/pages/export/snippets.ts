import type { InferenceInputSpec } from '@public/lib/tasks'

/**
 * curl and Python examples for POST /api/v1/predict/{runId}, shaped by the task's input:
 * a file upload, a JSON object of text fields, or a JSON feature record. Auth is a bearer API key.
 */
export function predictSnippets(runId: string, input: InferenceInputSpec, origin: string) {
  const url = `${origin}/api/v1/predict/${runId}`

  let curlBody: string
  let pyBody: string
  if (input.kind === 'file') {
    curlBody = `  -F "file=@input.${input.accept?.some((a) => a.startsWith('audio')) ? 'wav' : 'jpg'}"`
    pyBody = `    files={"file": open("input", "rb")},`
  } else if (input.kind === 'text') {
    const example = JSON.stringify(Object.fromEntries(input.fields.map((f) => [f, `your ${f} here`])))
    curlBody = `  -F 'fields=${example}'`
    pyBody = `    data={"fields": ${JSON.stringify(example)}},`
  } else {
    const example = '{"feature_a": 1.5, "feature_b": "value"}'
    curlBody = `  -F 'fields=${example}'`
    pyBody = `    data={"fields": ${JSON.stringify(example)}},`
  }

  const curl = `curl -X POST "${url}" \\
  -H "Authorization: Bearer $THESEUS_API_KEY" \\
${curlBody}`

  const python = `import os
import requests

response = requests.post(
    "${url}",
    headers={"Authorization": f"Bearer {os.environ['THESEUS_API_KEY']}"},
${pyBody}
)
response.raise_for_status()
print(response.json())`

  return { curl, python }
}

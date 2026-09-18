/**
 * README generation for export bundles. Kept as plain template-literal
 * functions rather than a file-based template — the content genuinely
 * branches on tier/lang, and this is just string interpolation, not a
 * templating language.
 */

import type { ExportFormat, ExportLang, ExportTier } from '@server/lib/enums'

export interface ReadmeVars {
  runName: string
  taskLabel: string
  format: ExportFormat
  tier: ExportTier
  hasVerify: boolean
}

/** Devkit langs ship source only (no project/build files) — dependencies + run commands live here instead. */
const DEVKIT_INFO: Record<
  'python' | 'typescript' | 'csharp' | 'java',
  {
    label: string
    files: string
    dependencies: string
    runCmd: string
    verifyCmd: string
    limitation: string
    note?: string
  }
> = {
  python: {
    label: 'Python',
    files: '`theseus_client.py`, `example.py`',
    dependencies: 'pip install onnxruntime numpy pillow',
    runCmd: 'python example.py <path-to-image>\npython example.py \'{"column": value, ...}\'   # tabular models',
    verifyCmd: 'python verify.py',
    limitation:
      "Implements preprocessing for **image classification/regression** and **tabular (numeric-only) inputs**. For other input types (text, audio), `preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`) — the client raises a clear error rather than silently producing wrong predictions; adapt the client's preprocessing method for your input type.",
  },
  typescript: {
    label: 'TypeScript (Bun)',
    files: '`client.ts`, `example.ts`',
    dependencies: 'bun add onnxruntime-node sharp',
    runCmd: 'bun example.ts <path-to-image>\nbun example.ts \'{"column": value, ...}\'   # tabular models',
    verifyCmd: 'bun verify.ts',
    limitation:
      "Implements preprocessing for **image classification/regression** and **tabular (numeric-only) inputs**. For other input types (text, audio), `preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`) — the client raises a clear error rather than silently producing wrong predictions; adapt the client's preprocessing method for your input type.",
  },
  csharp: {
    label: 'C#',
    files: '`TheseusClient.cs`, `Program.cs`',
    dependencies:
      'dotnet add package Microsoft.ML.OnnxRuntime --version 1.19.2\ndotnet add package SixLabors.ImageSharp --version 3.1.12',
    runCmd: 'dotnet run -- <path-to-image>',
    verifyCmd: 'dotnet run -- verify',
    limitation:
      "Only implements preprocessing for **image classification** inputs today. For other input types, `preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`) — the client raises a clear error rather than silently producing wrong predictions; adapt the client's preprocessing method for your input type.",
    note: '`Program.cs` uses top-level statements as its entry point — if your project already has one, merge its logic in rather than copying the file verbatim.',
  },
  java: {
    label: 'Java / Kotlin',
    files: '`TheseusClient.java`, `Main.java`',
    dependencies:
      '// Gradle (Kotlin DSL)\nimplementation("com.microsoft.onnxruntime:onnxruntime:1.29.0")\nimplementation("com.fasterxml.jackson.core:jackson-databind:2.22.1")',
    runCmd: 'java Main <path-to-image>',
    verifyCmd: 'java Main verify',
    limitation:
      "Only implements preprocessing for **image classification** inputs today. For other input types, `preprocessing.json` still has the exact parameters Ludwig used (`inputs[0].ludwigPreprocessing`) — the client raises a clear error rather than silently producing wrong predictions; adapt the client's preprocessing method for your input type.",
    note: 'Written in Java for maximum interop — call it directly from Kotlin with zero wrapping.',
  },
}

const APP_INFO: Record<'pwa' | 'flutter', { label: string; setupCmd: string; runCmd: string; verifyNote: string }> = {
  pwa: {
    label: 'Progressive Web App',
    setupCmd: 'python -m http.server 8000   # or: npx serve .',
    runCmd: 'Open http://localhost:8000 in a browser, then use the browser\'s "Install" prompt.',
    verifyNote: 'Use the "Run built-in self-check" button on the page.',
  },
  flutter: {
    label: 'Flutter app',
    setupCmd: 'flutter pub get',
    runCmd: 'flutter run   # or: flutter build apk / flutter build ios',
    verifyNote: 'Tap the checkmark icon in the app bar to run the built-in self-check screen.',
  },
}

export function renderReadme(vars: ReadmeVars, lang: ExportLang | null): string {
  const { runName, taskLabel, format, tier } = vars
  const header = `# ${runName}\n\nExported from Theseus — task: **${taskLabel}**, format: **${format}**.\n`

  if (tier === 'model') {
    return `${header}
## Contents

- \`model.${format}\` — the trained model artifact.
- \`preprocessing.json\` — the exact preprocessing Ludwig applied at training time (image resize/normalize, tokenizer params, ...) and, for classification outputs, the class list in the model's internal index order.
- \`labels.txt\` — one class name per line, in the same order as \`preprocessing.json\`'s output classes (if this is a classification task).

This is the bare artifact only — no client code. \`preprocessing.json\` has everything needed to reconstruct the input pipeline yourself; see the \`devkit\` export tier for a working reference implementation.
`
  }

  if (tier === 'app') {
    const info = APP_INFO[lang as 'pwa' | 'flutter']
    return `${header}
A ${info.label} — runs fully on-device (or in-browser), no server involved.

## Limitation

Only implements preprocessing for **image classification** inputs today. \`preprocessing.json\` still has the exact parameters Ludwig used (\`inputs[0].ludwigPreprocessing\`) for other input types — adapt the client code for your input type.

## Quick start

\`\`\`bash
${info.setupCmd}
\`\`\`

${info.runCmd}
${
  vars.hasVerify
    ? `\n## Verify\n\nThis bundle includes one real test-split sample and the platform's own prediction for it (\`expected.json\`). ${info.verifyNote} It confirms this app's own preprocessing reproduces that prediction within tolerance — the only proof in this bundle that its re-implementation of Ludwig's preprocessing is correct.\n`
    : ''
}`
  }

  // devkit
  const info = DEVKIT_INFO[lang as 'python' | 'typescript' | 'csharp' | 'java']
  return `${header}
Generated ${info.label} inference client — wraps the ONNX model with the same preprocessing/postprocessing Ludwig used at training time (see \`preprocessing.json\`). Ships as **source only** — no project/build file — so you can drop it straight into an existing project.

## Contents

${info.files}, \`preprocessing.json\`, \`labels.txt\`${vars.hasVerify ? ', `expected.json`, `sample/`' : ''}.

## Limitation

${info.limitation}

## Dependencies

\`\`\`
${info.dependencies}
\`\`\`
${info.note ? `\n${info.note}\n` : ''}
## Quick start

\`\`\`bash
${info.runCmd}
\`\`\`
${
  vars.hasVerify
    ? `\n## Verify\n\nThis bundle includes one real test-split sample and the platform's own prediction for it (\`expected.json\`). Run:\n\n\`\`\`bash\n${info.verifyCmd}\n\`\`\`\n\nto confirm this client's preprocessing reproduces that prediction within tolerance.\n`
    : ''
}`
}

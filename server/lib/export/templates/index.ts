/**
 * Export bundle template sources, imported as raw text so `bun build
 * --compile` inlines them into the binary (see server/types/templates.d.ts).
 * Templates are static — the only place `{{VAR}}` substitution happens is
 * README / manifest-ish files (package manifests, pubspec.yaml, HTML
 * titles); everything else (the clients themselves) reads preprocessing.json
 * / expected.json at runtime instead of being generated per-run.
 *
 * Devkit languages ship source only — no project/build files (no .csproj,
 * no Gradle, no package.json, no Docker). Required dependencies are
 * documented in the bundle's README for the user to add to their own
 * project. The `app` tier (pwa/flutter) is the exception: pubspec.yaml and
 * the PWA's index.html/manifest are the app's actual required entry points,
 * not incidental build scaffolding, so those stay.
 */

import programCs from './csharp/Program.cs' with { type: 'text' }
import theseusClientCs from './csharp/TheseusClient.cs' with { type: 'text' }
import flutterMain from './flutter/lib/main.dart.tmpl' with { type: 'text' }
import flutterClient from './flutter/lib/theseus_client.dart' with { type: 'text' }
import flutterPubspec from './flutter/pubspec.yaml.tmpl' with { type: 'text' }
import mainJava from './java/Main.java' with { type: 'text' }
import theseusClientJava from './java/TheseusClient.java' with { type: 'text' }
import pwaAppJs from './pwa/app.js' with { type: 'text' }
import pwaIcon from './pwa/icon.svg' with { type: 'text' }
import pwaIndexHtml from './pwa/index.html.tmpl' with { type: 'text' }
import pwaManifest from './pwa/manifest.webmanifest.tmpl' with { type: 'text' }
import pwaStyle from './pwa/style.css' with { type: 'text' }
import pwaServiceWorker from './pwa/sw.js' with { type: 'text' }
import examplePy from './python/example.py' with { type: 'text' }
import theseusClientPy from './python/theseus_client.py' with { type: 'text' }
import verifyPy from './python/verify.py' with { type: 'text' }
import clientTs from './typescript/client.ts.tmpl' with { type: 'text' }
import exampleTs from './typescript/example.ts.tmpl' with { type: 'text' }
import verifyTs from './typescript/verify.ts.tmpl' with { type: 'text' }

export const templates = {
  python: {
    client: theseusClientPy,
    example: examplePy,
    verify: verifyPy,
  },
  typescript: {
    client: clientTs,
    example: exampleTs,
    verify: verifyTs,
  },
  csharp: {
    client: theseusClientCs,
    program: programCs,
  },
  java: {
    client: theseusClientJava,
    main: mainJava,
  },
  pwa: {
    indexHtml: pwaIndexHtml,
    appJs: pwaAppJs,
    serviceWorker: pwaServiceWorker,
    manifest: pwaManifest,
    icon: pwaIcon,
    style: pwaStyle,
  },
  flutter: {
    pubspec: flutterPubspec,
    main: flutterMain,
    client: flutterClient,
  },
} as const

/** `{{VAR}}` substitution for README/manifest-ish files — deliberately not a template engine, just a literal replace. */
export function render(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => vars[key] ?? '')
}

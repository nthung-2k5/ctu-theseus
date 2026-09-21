# CLAUDE.md

Guidance for Claude Code sessions working in this repository. See `README.md` first for
architecture, NATS subjects, and S3 key layout — this file is about how to work in the repo, not
what it does.

## Verifying changes without running the stack

This repo is orchestrated by .NET Aspire (`apphost.mts`) and needs Postgres, NATS, and an
S3-compatible store running to actually execute end to end. Running the full stack (`aspire run`)
is expensive to set up and typically not available in an agent session — **don't assume you can
run it, and don't claim a change works end to end unless you've actually run it.** Default to
verifying with the checks below; if the user wants a live smoke test, they'll run `aspire run`
themselves or explicitly ask you to.

```bash
cd server && bun run typecheck              # tsc --noEmit
cd server && bun test                        # bun:test, server/**/*.test.ts
cd web    && bun run generate                  # Orval -> web/lib/api/generated (gitignored); needed before tsc on a fresh clone
cd web    && bunx tsc --noEmit -p tsconfig.app.json
cd web    && bun run build                   # tsc -b && vite build — the strongest single signal for web
cd ai_service && uv run pytest               # pytest, ai_service/tests/**
bunx biome check .                           # lint/format from repo root
```

A useful extra check for `server/` specifically: it's built with `bun build --compile`, which
inlines everything into a single binary. Anything read from disk at runtime (e.g.
`Bun.file('./templates/...')`) silently breaks under `--compile` even though it works fine under
`bun run --watch`. `server/lib/export/templates/` are imported as `with { type: 'text' }` for
exactly this reason — don't reintroduce filesystem template loading there.

## Conventions specific to this repo

- **No C#.** The Aspire AppHost is TypeScript (`apphost.mts`). If you're tempted to look for
  `Program.cs` or a `.csproj`, there isn't one.
- **Task registry is the source of truth.** `server/lib/tasks/registry.ts` drives the frontend
  task picker, per-modality dataset UI dispatch, snapshot column derivation, and Ludwig config
  compilation. Adding a trainable task is one registry entry, not five separate edits — check
  there first before assuming something needs a new file.
- **`server/`, `web/`, and the repo root are three independent Bun installs**, not a workspace.
  `web/lib/api.ts` has two `// @ts-expect-error` suppressions caused by this (TypeScript compares
  `elysia`'s `Elysia` class nominally because it has private/protected members, so the copy under
  `server/node_modules` and the copy under `web/node_modules` are mutually unassignable even when
  byte-identical). This is a known, deliberately-deferred issue — don't "fix" it opportunistically
  by casting elsewhere or deleting the suppressions; the real fix is turning this into a real Bun
  workspace with a pinned `elysia` version, which is out of scope unless asked for directly.
- **Cross-language schema.** `server/lib/schema.ts` is authoritative for NATS task/event payload
  shapes. After editing it, regenerate both sides:
  ```bash
  bun run server/compile_schema.ts   # from repo root; regenerates schema/*.json + ai_service/schema/*.py
  ```
  Review the diff — it should reflect only your intended schema change. This script needs to run
  with `ai_service`'s venv on `PATH` (via `uv run --project ai_service`) while writing to paths
  relative to the repo root; don't naively `cd` into `ai_service` to run it directly.
- **Class list correctness for exports.** Any code touching model export must source the label
  index order from `training_set_metadata.json`'s `idx2str` (written by Ludwig into
  `theseus-training/{runId}/results/**/model/`), never from the `label_classes` Postgres table.
  Postgres doesn't know what index Ludwig assigned to which class; getting this wrong produces a
  bundle that runs, returns confident predictions, and silently mislabels everything.
- **NATS event handlers must stay fast.** `server/lib/nats.ts` sets a short ack window; handlers
  in `server/lib/microservice.ts` flip a Postgres row's status and enqueue follow-up work, they
  don't do the follow-up work inline (e.g. export bundle assembly happens in a separate step, not
  in the `case 'export'` handler itself). At-least-once delivery means any DB write triggered by
  an event must be idempotent — the `UPDATE ... WHERE status='converting' RETURNING id` /
  zero-rows-means-skip pattern in `microservice.ts` is the template to follow for new event kinds.
- **Content-addressed storage.** Dataset pool uploads are deduped per-project by sha256
  (`uploadToPool` in `server/lib/storage.ts`). If you add a new upload path, reuse that helper
  rather than writing directly, or dedup silently breaks for that path.

## Git

Follow the standing Claude Code git safety rules (no `--force`, no `--no-verify`, no amending
someone else's commit, confirm before anything destructive). This repo has no repo-specific
overrides beyond that.

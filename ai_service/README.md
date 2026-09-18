# ai_service

The Python worker for CTU Theseus. Consumes NATS JetStream tasks dispatched by the gateway
(`server/`), runs [Ludwig](https://ludwig.ai) for AutoML training/export/inference, and
publishes progress events and artifacts back over NATS/S3. Never talks to Postgres directly —
all durable state lives in the gateway's database; this service only reads/writes S3 objects and
publishes events.

See the repo root `README.md` for the full architecture picture, NATS subject topology, and S3
key layout. This file covers just this package.

## Running

Managed by the Aspire AppHost (`../apphost.mts`, `addDockerfileBuilder('ai-worker',
'./ai_service', ...)`) — normally you don't run this directly, `aspire run` from the repo root
starts it alongside Postgres/NATS/RustFS/the gateway. If you need to run it standalone for
debugging:

```bash
cd ai_service
uv sync
uv run main.py
```

Requires `NATS_URI` (or Aspire's injected connection string env var), `S3_ENDPOINT`,
`S3_ACCESS_KEY`, `S3_SECRET_KEY`, and `PORT` (health endpoint) in the environment — Aspire injects
all of these automatically when launched via `aspire run`. `S3_ACCESS_KEY`/`S3_SECRET_KEY` must
match the gateway's (`server/lib/config.ts`); both default to `ctu-theseus`/`ctu-theseus-secret`
in development and are required outright when `ENVIRONMENT=production`.

## Layout

```
main.py              aiohttp health endpoint + NATS worker lifecycle
config.py             environment/config loading
constants.py          shared constants (bucket names, filenames — mirrors server/lib/config.ts)
telemetry.py           OpenTelemetry setup, trace context propagation over NATS headers

tasks/
  __init__.py          registers a consumer/subscription per subject (train/export/inference are
                        JetStream pull consumers; inference-warm is core NATS — see the
                        "Inference transport" note below)
  train.py             theseus.task.train.{runId} — compiles+runs a Ludwig training job
  export.py            theseus.task.export.{jobId} — converts a trained model to onnx/torchscript
  inference.py          theseus.task.inference.{inferenceId} — runs one inference job and
                        publishes its result; also theseus.inference.warm.{runId}
                        (fire-and-forget preload)

services/
  nats.py               connection setup, stream/consumer declarations, typed publish helpers
  storage.py             S3 helpers (download training inputs, upload results/artifacts)
  model_cache.py         in-process LRU cache of loaded LudwigModel instances, keyed by run id —
                          see "Inference transport" below
  predict.py             build_inference_output() — the tagged-union shape /api/inference
                          returns (classification/regression/text/tokens); parse_prediction_row()
                          — the flat {label: confidence} shape used only by export.py's
                          golden-sample verification step, kept separate since every generated
                          devkit/app client depends on that exact shape

schema/                 Pydantic models generated from server/lib/schema.ts — do not hand-edit,
                        regenerate via `bun run server/compile_schema.ts` from the repo root

tests/                  pytest; pythonpath=. is set in pyproject.toml so `from services.x import y`
                        resolves the same way it does at runtime (main.py's cwd is ai_service/)
```

## Key implementation notes

- **`training_set_metadata.json` is the source of truth for label indices.** Ludwig writes this
  alongside the trained model (`theseus-training/{runId}/results/**/model/`); its `idx2str` array
  is the only correct mapping from a model's output index to a class name. `inference.py` reads
  it for live inference; the gateway's export bundle assembly (`server/lib/export/metadata.ts`)
  reads the same file for generated-client label decoding. Never substitute the `label_classes`
  Postgres table for this — Postgres doesn't know Ludwig's internal index assignment.
- **`predict.py`'s two functions serve different consumers, deliberately.** `parse_prediction_row`
  turns raw Ludwig prediction output into a sorted `{label: confidence}` dict — used only for
  producing the `expected.json` golden sample that shipped devkit/app export bundles verify
  themselves against (every generated client, in every language, decodes that exact shape).
  `build_inference_output` is the live `/api/inference` response: a tagged union
  (`classification`/`regression`/`text`/`tokens`) since a flat `{label: confidence}` dict can't
  represent generated text or a token-tagged sequence. Don't merge them — changing
  `parse_prediction_row`'s shape breaks every previously-exported bundle's verify step.
- **Export golden sample.** `tasks/export.py` runs one real test-split row through the trained
  `LudwigModel.predict` after conversion and uploads it as `expected.json`
  (`theseus-models/{runId}/expected.json`). This is what the generated devkit/app bundle's
  `verify` script checks itself against — without it, a bug in the generated client's
  reimplemented preprocessing would ship silently.
- **Inference transport.** Inference is dispatched onto `THESEUS_TASKS` like train/export
  (`theseus.task.inference.{inferenceId}`, `subscribe_tasks` durable `"inference-worker"`) — not
  the core-NATS request/reply an earlier version used, which had no persistence, no redelivery,
  and a hard client-side timeout a cold model load routinely exceeded. `handle_inference_task`
  publishes the terminal result to `theseus.inference.result.{inferenceId}`
  (`THESEUS_INFERENCE_RESULTS`, `max_msgs_per_subject: 1`, 1-hour retention) on success;
  `on_inference_permanent_failure` does the same once `_consume_loop` exhausts retries, so a
  client polling `GET /api/inference/:runId/jobs/:inferenceId` never waits out the stream's full
  retention for nothing. There is deliberately no "pending"/"running" message — the poll route
  treats the absence of a result as pending. Because a task can be redelivered
  (`max_deliver=3`), a file-backed payload's upload is *not* deleted mid-attempt
  (`_resolve_file_input`); it's deleted once the outcome is final, in `handle_inference_task` on
  success or `on_inference_permanent_failure` after exhausted retries — deleting it eagerly would
  make a retry after a transient failure fail permanently with `FileNotFoundError`.

  Loaded models are cached in-process by `services/model_cache.py` (`model_cache`), an LRU
  bounded by `INFERENCE_MODEL_CACHE_SIZE` (`config.py`) — `LudwigModel.load()` deserializing the
  full checkpoint on every request was the dominant cost before this existed. The gateway fires
  `theseus.inference.warm.{runId}` (fire-and-forget core NATS, no reply, `handle_inference_warm`)
  when a user selects a model in the inference UI, to populate the cache before their first real
  request — this stays a request/reply-adjacent fire-and-forget signal rather than a dispatched
  task since there's no result to poll for.

## Testing

```bash
uv run pytest -v
```

Tests are pure-function unit tests — no NATS or S3, though the suite does import `torch` and
`ludwig` (via `tasks/inference` → `services/model_cache`), so the full dependency set has to be
installed to collect it. See `tests/test_predict.py` for the pattern. There is deliberately no integration test that spins up
a real Ludwig training run; that needs the full Aspire stack and is a manual smoke-test path
instead (see the repo root README).

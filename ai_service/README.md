# ai_service

The CTU Theseus backend: a single FastAPI service that serves the REST API, owns the Postgres
database, runs the durable job queue, and runs [Ludwig](https://ludwig.ai) for AutoML
training / export / inference. (It used to be only the Ludwig worker behind a separate Bun
gateway, joined by NATS. That gateway and NATS are gone.)

See the repo root `README.md` for the architecture picture (job queue, restart recovery, run
events, auth) and the S3 key layout. This file covers just this package.

## Running

Managed by the Aspire AppHost (`../apphost.mts`, `addDockerfileBuilder('api', './ai_service', ...)`).
Normally you don't run this directly: `aspire run` from the repo root starts it alongside
Postgres and RustFS. To run it standalone (needs Postgres 18 and an S3-compatible store):

```bash
cd ai_service
uv sync
uv run python main.py      # applies Alembic migrations, then serves on $PORT (default 8000)
```

Environment (Aspire injects all of these; `theseus/settings.py` is the reference):

| Variable | Purpose |
|---|---|
| `CTU_THESEUS_DB_URI` | Postgres connection string (`postgres://` is converted to `postgresql+asyncpg://`) |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` | object store |
| `S3_PUBLIC_ENDPOINT` | where the *browser* reaches the object store, used only to sign download URLs; defaults to `S3_ENDPOINT`. Needed when the API runs in a container (Aspire), where `S3_ENDPOINT` is a container-network hostname a browser cannot resolve |
| `JWT_SECRET` | signs access tokens |
| `ALLOWED_ORIGINS` | extra origins allowed to make cookie-authenticated writes |
| `ENVIRONMENT` | `production` requires the values above explicitly and makes cookies Secure |
| `COOKIE_SECURE`, `PORT`, `TEMP_DIR`, `INFERENCE_*`, `JOB_*`, `RUN_LOG_MAX_ROWS` | tuning |

**Run exactly one process.** No `--reload`, no `--workers`, no gunicorn: `theseus.lifespan`
refuses to start otherwise. Event fanout, the GPU lanes, the model cache, the rate limiter and the
abort registry all live in process memory.

## Layout

```
main.py                 migrations, then uvicorn
alembic.ini, migrations/  Alembic (async env); 0001_baseline is the whole schema
scripts/export_schema.py  writes/checks ../schema/openapi.json and ../schema/task_registry.json

theseus/
  app.py                create_app(): side-effect free (no DB, S3 or torch import), so the schema
                        export script and tests can import it
  lifespan.py           startup: single-process check, event writer, log handler, recovery,
                        dispatcher, reapers
  settings.py, constants.py, telemetry.py, metrics.py, errors.py, deps.py
  db/                   engine/session, enums, models (21 tables)
  auth/                 password (argon2id), JWT, refresh tokens, API keys, rate limiter
  routers/              one module per area; /api/v1 is the public API-key surface
  schemas/              pydantic request/response models (these are the OpenAPI contract)
  backends/             trainer backend plugins (base.py + registry.py, ludwig/); see "Writing a
                        backend" below
  services/             domain logic: task_registry (framework-neutral), snapshot (parquet), sweep,
                        datasets, training, inference, storage, evaluate, predict, model_cache, ...
  jobs/                 queue (claim/lease/CAS), dispatcher + lanes, train, export,
                        abort, recovery, reapers
  events/               single-writer run_events, in-process bus, SSE stream, log capture
  export/               bundle assembly, preprocessing decompiler, README generation, templates/

tests/                  pytest; DB tests need Postgres 18 and skip when it is unreachable
```

## Key implementation notes

- **The status column is the lock.** Every state transition is a guarded
  `UPDATE ... WHERE status = <expected> RETURNING`; zero rows means another actor got there first
  and the caller does nothing. Follow this for any new job kind or transition. Handlers must be
  idempotent for the same reason (startup recovery can re-queue a job whose first attempt already
  wrote something).
- **A trained model's own metadata is the source of truth for label indices**, never the
  `label_classes` Postgres table (it has no idea what index order a run assigned each class). For
  Ludwig this is `training_set_metadata.json`'s `idx2str` array, written under
  `theseus-training/{runId}/results/**/model/` and read by `LudwigLoadedModel` (live inference)
  and `backends/ludwig/manifest.py` (export bundle assembly, via `theseus/export/metadata.py`).
  Every `TrainerBackend.load()` must honor this the same way.
- **Only one thing inserts into `run_events`.** `events/writer.py` is the single writer, because a
  `BIGSERIAL` is not commit-ordered. Emitters (the training thread, request handlers, reapers)
  enqueue; the writer inserts and applies the projection in the same transaction.
- **Cancellation** is `cancel_requested_at` (durable) plus a `threading.Event` (fast path). Training
  raises `TrainingAborted(Exception)`, never `KeyboardInterrupt`.
- **A `LoadedModel`'s two prediction methods serve different consumers, deliberately.**
  `golden_prediction` yields the sorted `{label: confidence}` dict used only for the
  `expected.json` golden sample that every generated devkit/app client verifies itself against.
  `to_output` is the live inference response: a tagged union
  (`classification`/`regression`/`text`/`tokens`). Don't merge them: changing
  `golden_prediction`'s shape breaks every previously exported bundle's verify step.
- **Export golden sample.** After conversion the export job runs one real test-split row through the
  trained model and uploads it as `expected.json`; the bundle's verify script checks against it.
- **Inference.** A prediction (`POST /api/v1/predict/{runId}`, or the session route
  `/api/inference/{runId}`) runs inside the request and returns `{output}`; batch
  (`.../batch`) takes a CSV and returns the scored CSV with an `X-Row-Count` header. Nothing is queued
  or stored. Concurrency is capped at `inference_concurrency` (`503` + `Retry-After` when full) and
  the run time at `INFERENCE_TIMEOUT_SECONDS` (`504`); a thread cannot be interrupted, so a slot is
  only freed when its thread really finishes. Loaded models are cached in-process by
  `services/model_cache.py` (LRU, `INFERENCE_MODEL_CACHE_SIZE`), and `POST /api/inference/{runId}/warm`
  preloads one.
- **Templates are package data** (`theseus/export/templates/`), read from the installed package,
  and must be present in the Docker image (the Dockerfile copies the whole project).
- **Postgres generic plans.** After a prepared statement has run five times, Postgres may switch to
  a generic plan in which a bind parameter in an `ON CONFLICT ... WHERE` cannot match a partial
  unique index. The classify path therefore writes that predicate as a literal; keep it that way.

## Writing a trainer backend

A trainer backend is a package under `theseus/backends/` with one `TrainerBackend` subclass
(`theseus/backends/base.py`); `backends/ludwig/` is the reference implementation. Restart the
process and it shows up everywhere: `theseus.backends.registry.trainable_backends(task)` (which
tasks it can train), `GET /api/projects/{id}/training-backends` (its models and hyperparameters,
for the create-run/sweep UI), and every run trained with it (`training_runs.backend`) flows
through the rest of the pipeline — `jobs/train.py`, `services/model_cache.py`,
`services/inference.py`, `jobs/export.py` — with no backend-specific code outside the package itself.

What to implement, roughly in the order a run touches them:

1. `Hyperparameters`: a pydantic model (subclass `backends.base.HyperparamsBase`) for the knobs
   your models accept. Unknown keys are rejected, so a client that sends a knob you don't have
   gets a clear 422, not a silently ignored field.
2. `supports(task)` / `models(task)`: which tasks you can train, and (per task) the selectable
   models/architectures — what used to be a fixed encoder list.
3. `compile(task, ctx, hp)`: turn a task descriptor, snapshot context and validated
   hyperparameters into your own config dict. It's stored verbatim on the run
   (`training_runs.config`) and never interpreted outside your package.
4. `train(run)`: train synchronously against `run.dataset_uri` / `run.output_uri` (s3fs paths),
   calling `run.report(epoch, split, metrics)` as you go and `run.check_abort()` wherever it's
   safe to stop. Report a `validation`-split `loss` if you have one — it drives the run's
   `best_epoch`. Return the trained model already loaded (see `load` below).
5. `load(model_dir)`: load a trained model from a downloaded directory into a `LoadedModel`
   (`predict`, `to_output`, `golden_prediction`, `close`) — the interface `model_cache.py`,
   `services/inference.py` and `jobs/export.py` actually talk to.
6. `evaluate(model, df, split_column, item_id_column)` (optional) and `convert(model, artifact_id,
   workdir)` / `artifacts` (optional): the evaluation report and export conversions, if you want
   either. Both default to "not supported" so a minimal backend can skip them.

**Import discipline matters more here than almost anywhere else in the codebase.** Listing
installed backends (`trainable_backends`, the `training-backends` endpoint, even `create_app()`
via `scripts/export_schema.py`) must not require your ML framework to be importable — the process
that only serves the API and the one that trains on a GPU are the same process. Your package's own
`__init__.py` and any module it imports at top level must therefore be free of `import torch` /
`import ludwig` / etc.; do those imports inside the classmethod bodies that need them (`train`,
`load`, `evaluate`, `convert`), the way `backends/ludwig/__init__.py` does. `available()` should
try the import and return a user-facing reason on failure, for an optional-dependency backend.

## Testing

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python scripts/export_schema.py --check     # committed OpenAPI/task-registry are current
```

The suite runs against a real Postgres 18 (`TEST_DATABASE_URI`, default
`postgres://theseus:theseus@127.0.0.1:55432/theseus_test`; the test database is dropped and
recreated). Use `127.0.0.1`, not `localhost`, on Windows (IPv6 makes asyncpg hang). S3, Ludwig and
the GPU are faked, so a real training run, real S3 and CUDA paths remain a manual smoke test via
`aspire run`.

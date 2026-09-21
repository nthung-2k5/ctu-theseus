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
  services/             domain logic: task_registry, ludwig_config, snapshot (parquet), sweep,
                        datasets, training, inference, storage, evaluate, predict, model_cache, ...
  jobs/                 queue (claim/lease/CAS), dispatcher + lanes, train, export, inference,
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
- **`training_set_metadata.json` is the source of truth for label indices.** Ludwig writes it
  under `theseus-training/{runId}/results/**/model/`; its `idx2str` array is the only correct
  mapping from a model's output index to a class name. Live inference and export bundle assembly
  (`theseus/export/metadata.py`) both read it. Never substitute the `label_classes` table:
  Postgres doesn't know the index Ludwig assigned, and getting this wrong yields a bundle that
  runs, predicts confidently, and mislabels everything.
- **Only one thing inserts into `run_events`.** `events/writer.py` is the single writer, because a
  `BIGSERIAL` is not commit-ordered. Emitters (the training thread, request handlers, reapers)
  enqueue; the writer inserts and applies the projection in the same transaction.
- **Cancellation** is `cancel_requested_at` (durable) plus a `threading.Event` (fast path). Training
  raises `TrainingAborted(Exception)`, never `KeyboardInterrupt`.
- **`predict.py`'s two functions serve different consumers, deliberately.** `parse_prediction_row`
  yields the sorted `{label: confidence}` dict used only for the `expected.json` golden sample that
  every generated devkit/app client verifies itself against. `build_inference_output` is the live
  inference response: a tagged union (`classification`/`regression`/`text`/`tokens`). Don't merge
  them: changing `parse_prediction_row`'s shape breaks every previously exported bundle's verify step.
- **Export golden sample.** After conversion the export job runs one real test-split row through the
  trained model and uploads it as `expected.json`; the bundle's verify script checks against it.
- **Inference.** Sync predict (`/api/v1/predict/{runId}/sync`) awaits a shielded job with a bounded
  wait and falls back to `202 + inferenceId` on timeout (the job keeps running). Async and batch
  inputs go to the `theseus-uploads` bucket and are deleted once the job ends. Loaded models are
  cached in-process by `services/model_cache.py` (LRU, `INFERENCE_MODEL_CACHE_SIZE`), and
  `POST /api/inference/{runId}/warm` preloads one.
- **Templates are package data** (`theseus/export/templates/`), read from the installed package,
  and must be present in the Docker image (the Dockerfile copies the whole project).
- **Postgres generic plans.** After a prepared statement has run five times, Postgres may switch to
  a generic plan in which a bind parameter in an `ON CONFLICT ... WHERE` cannot match a partial
  unique index. The classify path therefore writes that predicate as a literal; keep it that way.

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

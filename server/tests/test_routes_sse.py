"""GET /api/runs/{id}/events against a REAL uvicorn server.

httpx in-process transports buffer whole responses, so they cannot prove streaming. This starts
the app on an ephemeral port and reads the live stream.
"""

import asyncio
import contextlib
import json
import uuid

import asyncpg
import httpx
import pytest
import sqlalchemy as sa
import uvicorn
from conftest import TEST_DB_URI, _split

from theseus.db.models import DatasetVersion, TrainingRun
from theseus.events import get_event_bus, get_event_writer


@pytest.fixture
async def base_url(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    server.capture_signals = contextlib.nullcontext  # do not hijack pytest's signal handlers
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await asyncio.wait_for(task, 10)


async def signed_up(base_url):
    c = httpx.AsyncClient(base_url=base_url, timeout=10)
    email = f"{uuid.uuid4().hex[:8]}@example.com"
    r = await c.post("/api/auth/register", json={"name": "U", "email": email, "password": "correct horse battery"})
    assert r.status_code == 201
    return c, uuid.UUID(r.json()["user"]["id"])


async def make_run_for(c, db):
    p = (await c.post("/api/projects", json={"name": "p", "description": None, "task": "image_classification"})).json()
    project_id = uuid.UUID(p["project"]["id"])
    async with db() as s:
        version_id = (
            await s.execute(sa.select(DatasetVersion.id).where(DatasetVersion.dataset_id == project_id))
        ).scalar_one()
        run = TrainingRun(
            project_id=project_id, dataset_version_id=version_id, name="r", status="running", hyperparameters={}
        )
        s.add(run)
        await s.commit()
        return run.id


async def read_events(lines, until, timeout=10):
    """Parse SSE frames from one live line iterator until `until(frame)` is true.

    Comment / retry-only frames (no `event:` or `data:`) are not events: they are attached to the
    next real frame as `_comments`.
    """
    frames: list[dict] = []
    current: dict = {}
    pending_comments: list[str] = []

    async def consume():
        async for line in lines:
            if line == "":
                if not current:
                    continue
                if "event" in current or "data" in current:
                    if pending_comments:
                        current["_comments"] = list(pending_comments)
                        pending_comments.clear()
                    frames.append(dict(current))
                    current.clear()
                    if until(frames[-1]):
                        return
                else:
                    pending_comments.extend(current.pop("_comments", []))
                    current.clear()
            elif line.startswith(":") or line.startswith("retry:"):
                current.setdefault("_comments", []).append(line)
            else:
                key, _, value = line.partition(": ")
                current[key] = value

    await asyncio.wait_for(consume(), timeout)
    return frames


async def stray_transactions() -> int:
    admin, name = _split(TEST_DB_URI)
    conn = await asyncpg.connect(admin)
    try:
        rows = await conn.fetch(
            "SELECT pid, left(query, 200) AS q FROM pg_stat_activity WHERE datname = $1 AND state = 'idle in transaction'",  # noqa: E501
            name,
        )
        return len(rows)
    finally:
        await conn.close()


async def test_events_stream_live_replay_from_last_event_id_and_release_the_db_connection(base_url, db):
    c, _ = await signed_up(base_url)
    run_id = await make_run_for(c, db)
    writer = get_event_writer()
    writer.status(run_id, "running")
    writer.metric(run_id, 1, "validation", {"loss": 0.9})
    await writer.flush()

    async with c.stream("GET", f"/api/runs/{run_id}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "no-cache" in r.headers["cache-control"]

        lines = r.aiter_lines()  # httpx allows exactly one iterator per response, so reuse it across reads
        backlog = await read_events(lines, until=lambda f: f.get("event") == "metric")
        assert [f["event"] for f in backlog] == ["status", "metric"]
        # The stream can live for hours, so once the backlog is delivered it must NOT be pinning a
        # pooled DB connection in a transaction. (Asserted after the backlog read, because the
        # response headers arrive before the body's own backlog query has finished.)
        assert await stray_transactions() == 0
        assert backlog[0]["_comments"] == ["retry: 3000"]  # reconnect hint on the first frame
        assert json.loads(backlog[1]["data"]) == {
            "kind": "metric", "runId": str(run_id), "ts": json.loads(backlog[1]["data"])["ts"],
            "epoch": 1, "split": "validation", "metrics": {"loss": 0.9},
        }  # fmt: skip
        first_id, second_id = int(backlog[0]["id"]), int(backlog[1]["id"])
        assert second_id > first_id

        # A live event, emitted after the client is already connected and reading.
        writer.metric(run_id, 2, "validation", {"loss": 0.4})
        live = await read_events(lines, until=lambda f: f.get("event") == "metric")
        assert json.loads(live[-1]["data"])["epoch"] == 2 and int(live[-1]["id"]) > second_id

    # Reconnect the way EventSource does: it sends the last id it saw and gets only what it missed.
    async with c.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(second_id)}) as r:
        writer.status(run_id, "succeeded")
        frames = await read_events(r.aiter_lines(), until=lambda f: f.get("event") == "status")
    assert [json.loads(f["data"]).get("epoch") or json.loads(f["data"]).get("status") for f in frames] == [
        2,
        "succeeded",
    ]

    for _ in range(50):  # the disconnect must unsubscribe, or every closed tab would leak a queue
        if get_event_bus().subscriber_count(str(run_id)) == 0:
            break
        await asyncio.sleep(0.1)
    assert get_event_bus().subscriber_count(str(run_id)) == 0
    await c.aclose()


async def test_another_users_stream_is_forbidden_and_an_unknown_run_is_404(base_url, db):
    owner, _ = await signed_up(base_url)
    stranger, _ = await signed_up(base_url)
    run_id = await make_run_for(owner, db)
    assert (await stranger.get(f"/api/runs/{run_id}/events")).status_code == 403
    assert (await owner.get(f"/api/runs/{uuid.uuid4()}/events")).status_code == 404
    assert (await httpx.AsyncClient(base_url=base_url).get(f"/api/runs/{run_id}/events")).status_code == 401
    await owner.aclose()
    await stranger.aclose()

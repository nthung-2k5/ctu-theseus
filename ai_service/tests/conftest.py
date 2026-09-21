"""Shared fixtures. DB-backed tests need a real Postgres 18 (uuidv7()) and skip cleanly without one.

Point TEST_DATABASE_URI at any Postgres 18 server; the test database is dropped and recreated.
"""

import asyncio
import os
from urllib.parse import urlparse, urlunparse

import asyncpg
import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

TEST_DB_URI = os.environ.get("TEST_DATABASE_URI", "postgres://theseus:theseus@127.0.0.1:55432/theseus_test")


def _split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    admin = urlunparse(parsed._replace(path="/postgres"))
    return admin, parsed.path.lstrip("/")


@pytest.fixture(scope="session")
def database_url() -> str:
    import theseus.db.models  # noqa: F401
    from theseus.db.base import Base

    admin, name = _split(TEST_DB_URI)
    sa_url = "postgresql+asyncpg://" + TEST_DB_URI.split("://", 1)[1]

    async def setup() -> None:
        conn = await asyncpg.connect(admin, timeout=3)
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await conn.execute(f'CREATE DATABASE "{name}"')
        finally:
            await conn.close()
        engine = create_async_engine(sa_url, poolclass=NullPool)
        async with engine.begin() as c:
            await c.run_sync(Base.metadata.create_all)
        await engine.dispose()

    try:
        asyncio.run(setup())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable for DB tests: {e}")
    return sa_url


async def _kill_stray_connections() -> None:
    """Terminate every other backend on the test database.

    A task cancelled mid-query (dispatcher shutdown tests) can abandon a connection that is
    idle in a transaction. It holds locks that would make the next test TRUNCATE wait forever,
    so treat any leftover connection as garbage.
    """
    admin, name = _split(TEST_DB_URI)
    conn = await asyncpg.connect(admin, timeout=3)
    try:
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
            name,
        )
    finally:
        await conn.close()


@pytest.fixture
async def db(database_url, monkeypatch):
    """A clean database per test, wired into theseus.db.base so app code uses it."""
    from theseus.db import base

    await _kill_stray_connections()
    engine = create_async_engine(database_url, poolclass=NullPool)
    monkeypatch.setattr(base, "_engine", engine)
    monkeypatch.setattr(base, "_sessionmaker", async_sessionmaker(engine, expire_on_commit=False))
    async with engine.begin() as conn:
        tables = ", ".join(f'"{t.name}"' for t in base.Base.metadata.sorted_tables)
        await conn.execute(sa.text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    yield base._sessionmaker
    await engine.dispose()
    await _kill_stray_connections()


@pytest.fixture
async def app(db):
    """The real app with a started event writer (routes that cancel or emit need one)."""
    from theseus.app import create_app
    from theseus.events import InProcessRunEventBus, set_event_bus, set_event_writer
    from theseus.events.writer import EventWriter

    bus = InProcessRunEventBus()
    set_event_bus(bus)
    writer = EventWriter(bus)
    await writer.start()
    set_event_writer(writer)
    yield create_app()
    await writer.stop()
    set_event_writer(None)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def new_user(app):
    """Factory: a separate HTTP client (own cookies) already registered as a fresh user."""
    clients: list[httpx.AsyncClient] = []

    async def _make(email: str | None = None) -> httpx.AsyncClient:
        import uuid

        c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clients.append(c)
        r = await c.post(
            "/api/auth/register",
            json={
                "name": "User",
                "email": email or f"{uuid.uuid4().hex[:8]}@example.com",
                "password": "correct horse battery",
            },
        )
        assert r.status_code == 201, r.text
        c.user_id = r.json()["user"]["id"]  # type: ignore[attr-defined]
        return c

    yield _make
    # closed by the event loop teardown; clients hold no external resources


@pytest.fixture
def make_run(db):
    """Create a project + snapshot + training run, returning the run id."""
    import uuid

    from theseus.db.models import Dataset, DatasetVersion, Project, TrainingRun, User

    async def _make(status: str = "queued", task: str = "image_classification", **run_fields):
        async with db() as s:
            user = User(name="U", email=f"{uuid.uuid4()}@x.co", password_hash="x")
            s.add(user)
            await s.flush()
            project = Project(user_id=user.id, name="p", task=task)
            s.add(project)
            await s.flush()
            s.add(Dataset(project_id=project.id, modality="vision"))
            await s.flush()
            version = DatasetVersion(dataset_id=project.id, version_tag="v1", status="ready")
            s.add(version)
            await s.flush()
            run = TrainingRun(
                project_id=project.id,
                dataset_version_id=version.id,
                name="run",
                status=status,
                hyperparameters={},
                config_key="cfg",
                **run_fields,
            )
            s.add(run)
            await s.commit()
            return run.id

    return _make


@pytest.fixture
def make_export(db, make_run):
    """Create an export row (and its run), returning (export_id, run_id)."""
    from theseus.db.models import ModelExport, TrainingRun

    async def _make(status: str = "pending", tier: str = "model", fmt: str = "onnx", run_id=None, **fields):
        import sqlalchemy as sa

        rid = run_id or await make_run(status="succeeded")
        async with db() as s:
            user_id = (await s.execute(sa.select(TrainingRun.project_id).where(TrainingRun.id == rid))).scalar_one()
            from theseus.db.models import Project

            owner = (await s.execute(sa.select(Project.user_id).where(Project.id == user_id))).scalar_one()
            e = ModelExport(run_id=rid, user_id=owner, tier=tier, format=fmt, status=status, **fields)
            s.add(e)
            await s.commit()
            return e.id, rid

    return _make


@pytest.fixture
def make_inference(db, make_run):
    """Create an inference job row, returning (job_id, run_id)."""
    from theseus.db.models import InferenceJob

    async def _make(status: str = "pending", payload=None, run_id=None, **fields):
        rid = run_id or await make_run(status="succeeded")
        async with db() as s:
            job = InferenceJob(
                run_id=rid, status=status, payload=payload or {"kind": "text", "fields": {"text": "hi"}}, **fields
            )
            s.add(job)
            await s.commit()
            return job.id, rid

    return _make

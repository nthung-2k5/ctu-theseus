"""Startup seeds an administrator only when the database has none."""

import pytest
import sqlalchemy as sa

from omnitrain.db.models import User
from omnitrain.services.bootstrap_admin import ensure_admin
from omnitrain.settings import get_settings

EMAIL = "root@example.com"
PASSWORD = "seeded password 1"


@pytest.fixture
def seed_env(monkeypatch):
    monkeypatch.setenv("OMNITRAIN_ADMIN_EMAIL", f" {EMAIL.upper()} ")
    monkeypatch.setenv("OMNITRAIN_ADMIN_PASSWORD", PASSWORD)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _users(db) -> list[User]:
    async with db() as s:
        return list((await s.execute(sa.select(User))).scalars())


async def test_creates_an_admin_who_can_sign_in(seed_env, db, client):
    await ensure_admin()

    [user] = await _users(db)
    assert (user.email, user.role) == (EMAIL, "admin")
    r = await client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 200
    assert r.json()["user"]["role"] == "admin"


async def test_runs_again_without_creating_a_second_account(seed_env, db):
    await ensure_admin()
    await ensure_admin()
    assert len(await _users(db)) == 1


async def test_does_nothing_when_an_admin_already_exists(seed_env, db, new_user, monkeypatch):
    monkeypatch.setenv("OMNITRAIN_ADMIN_EMAILS", "someone@example.com")
    get_settings.cache_clear()
    await new_user("someone@example.com")  # promoted to admin by the bootstrap list

    await ensure_admin()

    assert [u.email for u in await _users(db)] == ["someone@example.com"]


async def test_promotes_an_ordinary_account_that_holds_the_address(seed_env, db, new_user):
    await new_user(EMAIL)
    await ensure_admin()

    [user] = await _users(db)
    assert user.role == "admin"


async def test_development_falls_back_to_the_dev_password(db, client, monkeypatch):
    monkeypatch.delenv("OMNITRAIN_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("OMNITRAIN_ADMIN_EMAIL", raising=False)
    get_settings.cache_clear()
    try:
        await ensure_admin()
        s = get_settings()
        r = await client.post("/api/auth/login", json={"email": s.admin_email, "password": s.dev_admin_password})
        assert r.status_code == 200
    finally:
        get_settings.cache_clear()


async def test_production_without_a_password_seeds_nothing(db, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("OMNITRAIN_ADMIN_PASSWORD", raising=False)
    for name, value in {
        "CTU_OMNITRAIN_DB_URI": "postgres://x",
        "S3_ENDPOINT": "http://x",
        "S3_ACCESS_KEY": "x",
        "S3_SECRET_KEY": "x",
        "JWT_SECRET": "x",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    try:
        await ensure_admin()
        assert await _users(db) == []
    finally:
        get_settings.cache_clear()

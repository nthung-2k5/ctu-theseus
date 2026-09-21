import uuid

import pytest
import sqlalchemy as sa

from theseus.auth import rate_limit
from theseus.auth.api_keys import KEY_PREFIX, generate_api_key, hash_api_key
from theseus.auth.jwt import InvalidToken, create_access_token, decode_access_token
from theseus.db.models import Project, RefreshToken, User
from theseus.lifespan import assert_single_process

CREDS = {"name": "Ada", "email": "Ada@Example.com", "password": "correct horse battery"}


# -- Pure unit tests (no DB) ------------------------------------------------------------------


def test_access_token_roundtrip_and_tamper_rejection():
    uid = uuid.uuid4()
    token = create_access_token(uid)
    assert decode_access_token(token) == uid
    with pytest.raises(InvalidToken):
        decode_access_token(token[:-2] + "xx")
    with pytest.raises(InvalidToken):
        decode_access_token("not-a-jwt")


def test_api_key_format_and_hash_are_deterministic():
    raw, key_hash, prefix = generate_api_key()
    assert raw.startswith(KEY_PREFIX) and len(raw) == len(KEY_PREFIX) + 48
    assert hash_api_key(raw) == key_hash and len(key_hash) == 64
    assert raw.startswith(prefix) and prefix != raw
    assert generate_api_key()[0] != raw


def test_rate_limit_blocks_after_max_and_resets_after_window():
    rate_limit.reset()
    results = [rate_limit.check("k", 3, 60, now=0.0) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[3].retry_after >= 1
    assert rate_limit.check("other", 3, 60, now=0.0).allowed  # independent per key
    assert rate_limit.check("k", 3, 60, now=61.0).allowed  # new window


def test_rate_limit_sweep_drops_only_expired_windows():
    rate_limit.reset()
    rate_limit.check("old", 1, 60, now=0.0)
    rate_limit.check("new", 1, 60, now=100.0)
    assert rate_limit.sweep(60, now=120.0) == 1


@pytest.mark.parametrize("argv", [["--workers", "2"], ["--workers=4"], ["--reload"]])
def test_single_process_assertion_rejects_multi_worker_and_reload(argv):
    with pytest.raises(RuntimeError, match="single process"):
        assert_single_process(argv)


def test_single_process_assertion_allows_default_and_one_worker():
    assert_single_process([])
    assert_single_process(["--workers", "1"])


def test_operation_ids_are_unique():
    from theseus.app import create_app

    ids = [op["operationId"] for path in create_app().openapi()["paths"].values() for op in path.values()]
    assert len(ids) == len(set(ids))


# -- HTTP tests (real Postgres) ---------------------------------------------------------------


async def test_register_sets_cookies_and_normalizes_email(client, db):
    r = await client.post("/api/auth/register", json=CREDS)
    assert r.status_code == 201
    assert r.json()["user"]["email"] == "ada@example.com"
    assert "passwordHash" not in r.text and "password_hash" not in r.text
    set_cookie = "\n".join(r.headers.get_list("set-cookie")).lower()
    assert "access_token=" in set_cookie and "refresh_token=" in set_cookie
    assert "httponly" in set_cookie
    assert "path=/api/auth" in set_cookie
    async with db() as s:
        assert (await s.execute(sa.select(User.password_hash))).scalar_one().startswith("$argon2id$")


async def test_duplicate_email_is_409_case_insensitively(client):
    assert (await client.post("/api/auth/register", json=CREDS)).status_code == 201
    r = await client.post("/api/auth/register", json={**CREDS, "email": "ADA@example.com"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "CONFLICT"


async def test_register_validation_uses_error_envelope(client):
    r = await client.post("/api/auth/register", json={**CREDS, "password": "short"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION"


async def test_me_requires_a_session_then_returns_the_user(client):
    assert (await client.get("/api/auth/me")).status_code == 401
    await client.post("/api/auth/register", json=CREDS)
    r = await client.get("/api/auth/me")
    assert r.status_code == 200 and r.json()["user"]["name"] == "Ada"


async def test_login_rejects_bad_password_and_unknown_email_identically(client):
    await client.post("/api/auth/register", json=CREDS)
    client.cookies.clear()
    wrong = await client.post("/api/auth/login", json={"email": CREDS["email"], "password": "nope-nope-nope"})
    unknown = await client.post("/api/auth/login", json={"email": "who@example.com", "password": "nope-nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    ok = await client.post("/api/auth/login", json={"email": CREDS["email"], "password": CREDS["password"]})
    assert ok.status_code == 200
    assert (await client.get("/api/auth/me")).status_code == 200


async def test_refresh_rotates_and_reuse_revokes_the_whole_family(client, db):
    await client.post("/api/auth/register", json=CREDS)
    old_refresh = client.cookies.get("refresh_token", path="/api/auth")
    assert old_refresh

    r = await client.post("/api/auth/refresh")
    assert r.status_code == 204
    new_refresh = client.cookies.get("refresh_token", path="/api/auth")
    assert new_refresh and new_refresh != old_refresh

    # Replaying the already-rotated token is treated as theft: the whole family dies.
    client.cookies.set("refresh_token", old_refresh, path="/api/auth")
    assert (await client.post("/api/auth/refresh")).status_code == 401
    client.cookies.set("refresh_token", new_refresh, path="/api/auth")
    assert (await client.post("/api/auth/refresh")).status_code == 401
    async with db() as s:
        live = (await s.execute(sa.select(sa.func.count()).where(RefreshToken.revoked_at.is_(None)))).scalar_one()
        assert live == 0


async def test_logout_revokes_refresh_token(client):
    await client.post("/api/auth/register", json=CREDS)
    refresh = client.cookies.get("refresh_token", path="/api/auth")
    assert (await client.post("/api/auth/logout")).status_code == 204
    client.cookies.set("refresh_token", refresh, path="/api/auth")
    assert (await client.post("/api/auth/refresh")).status_code == 401


async def test_cross_origin_writes_are_blocked_but_same_origin_and_no_origin_pass(client):
    blocked = await client.post(
        "/api/auth/login", json={"email": "a@b.co", "password": "x"}, headers={"Origin": "https://evil.example"}
    )
    assert blocked.status_code == 403
    same = await client.post(
        "/api/auth/login", json={"email": "a@b.co", "password": "x"}, headers={"Origin": "http://test"}
    )
    assert same.status_code == 401  # got past the origin check, failed on credentials
    none = await client.post("/api/auth/login", json={"email": "a@b.co", "password": "x"})
    assert none.status_code == 401


async def test_ownership_dependency_404s_and_403s(client, db):
    from fastapi import APIRouter

    from theseus.app import create_app
    from theseus.deps import ProjectDep

    probe = APIRouter()

    @probe.get("/api/_probe/{project_id}")
    async def _probe(project: ProjectDep) -> dict[str, str]:
        return {"id": str(project.id)}

    app = create_app()
    app.include_router(probe)

    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/api/auth/register", json=CREDS)
        me = (await c.get("/api/auth/me")).json()["user"]["id"]
        async with db() as s:
            other = User(name="Bob", email="bob@example.com", password_hash="x")
            s.add(other)
            await s.flush()
            mine = Project(user_id=uuid.UUID(me), name="mine", task="image_classification")
            theirs = Project(user_id=other.id, name="theirs", task="image_classification")
            s.add_all([mine, theirs])
            await s.commit()
            mine_id, theirs_id = mine.id, theirs.id

        assert (await c.get(f"/api/_probe/{mine_id}")).json() == {"id": str(mine_id)}
        assert (await c.get(f"/api/_probe/{theirs_id}")).status_code == 403
        assert (await c.get(f"/api/_probe/{uuid.uuid4()}")).status_code == 404

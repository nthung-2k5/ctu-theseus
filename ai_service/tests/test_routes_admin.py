"""/api/admin: the admin guard, bootstrap promotion, and user management."""

import pytest

from theseus.settings import get_settings

ADMIN_EMAIL = "boss@example.com"


@pytest.fixture
def admin_emails(monkeypatch):
    monkeypatch.setenv("THESEUS_ADMIN_EMAILS", f" {ADMIN_EMAIL.upper()} , other@example.com")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_non_admin_gets_403(new_user):
    user = await new_user()
    assert (await user.get("/api/admin/users")).status_code == 403


async def test_anonymous_gets_401(client):
    assert (await client.get("/api/admin/users")).status_code == 401


async def test_bootstrap_email_is_promoted_on_register(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    me = (await admin.get("/api/auth/me")).json()["user"]
    assert me["role"] == "admin"
    assert (await admin.get("/api/admin/users")).status_code == 200


async def test_bootstrap_email_is_promoted_on_login(new_user, client, monkeypatch):
    await new_user(ADMIN_EMAIL)  # registered before the email was listed: a plain user
    assert (
        await client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "correct horse battery"})
    ).json()["user"]["role"] == "user"

    monkeypatch.setenv("THESEUS_ADMIN_EMAILS", ADMIN_EMAIL)
    get_settings.cache_clear()
    try:
        r = await client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "correct horse battery"})
        assert r.json()["user"]["role"] == "admin"
    finally:
        get_settings.cache_clear()


async def test_list_search_and_counts(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    await new_user("alice@example.com")
    await new_user("bob@example.com")

    everyone = (await admin.get("/api/admin/users")).json()
    assert everyone["total"] == 3

    found = (await admin.get("/api/admin/users", params={"q": "ALICE"})).json()
    assert [u["email"] for u in found["users"]] == ["alice@example.com"]
    assert found["users"][0]["projectCount"] == 0

    admins = (await admin.get("/api/admin/users", params={"role": "admin"})).json()
    assert [u["email"] for u in admins["users"]] == [ADMIN_EMAIL]

    # A LIKE wildcard in the search text is matched literally, not as a pattern.
    assert (await admin.get("/api/admin/users", params={"q": "%"})).json()["total"] == 0


async def test_promote_and_demote(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    other = await new_user("alice@example.com")

    r = await admin.patch(f"/api/admin/users/{other.user_id}", json={"role": "admin"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "admin"
    assert (await other.get("/api/admin/users")).status_code == 200

    r = await admin.patch(f"/api/admin/users/{other.user_id}", json={"role": "user"})
    assert r.json()["user"]["role"] == "user"
    assert (await other.get("/api/admin/users")).status_code == 403


async def test_cannot_demote_or_disable_yourself(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    assert (await admin.patch(f"/api/admin/users/{admin.user_id}", json={"role": "user"})).status_code == 400
    assert (await admin.patch(f"/api/admin/users/{admin.user_id}", json={"disabled": True})).status_code == 400


async def test_cannot_remove_the_last_active_admin(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    second = await new_user("other@example.com")  # also a bootstrap admin
    assert (await admin.patch(f"/api/admin/users/{second.user_id}", json={"disabled": True})).status_code == 200
    # `second` is now disabled, so `admin` is the only active admin left; nobody else can remove them,
    # and they cannot remove themselves.
    assert (await admin.patch(f"/api/admin/users/{admin.user_id}", json={"role": "user"})).status_code == 400


async def test_disabled_user_cannot_sign_in_or_refresh(admin_emails, new_user, client):
    admin = await new_user(ADMIN_EMAIL)
    victim = await new_user("alice@example.com")

    r = await admin.patch(f"/api/admin/users/{victim.user_id}", json={"disabled": True})
    assert r.json()["user"]["disabled"] is True

    login = await client.post(
        "/api/auth/login", json={"email": "alice@example.com", "password": "correct horse battery"}
    )
    assert login.status_code == 403
    assert (await victim.post("/api/auth/refresh")).status_code == 401
    # The still-valid access token does not open the admin area either way.
    assert (await victim.get("/api/admin/users")).status_code in (401, 403)

    r = await admin.patch(f"/api/admin/users/{victim.user_id}", json={"disabled": False})
    assert r.json()["user"]["disabled"] is False
    login = await client.post(
        "/api/auth/login", json={"email": "alice@example.com", "password": "correct horse battery"}
    )
    assert login.status_code == 200


async def test_revoke_sessions_and_api_keys(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    victim = await new_user("alice@example.com")
    key = (await victim.post("/api/keys", json={"name": "ci"})).json()

    listed = (await admin.get(f"/api/admin/users/{victim.user_id}/api-keys")).json()["keys"]
    assert [k["id"] for k in listed] == [key["id"]]

    assert (await admin.post(f"/api/admin/users/{victim.user_id}/revoke-sessions")).status_code == 204
    assert (await victim.post("/api/auth/refresh")).status_code == 401

    assert (await admin.delete(f"/api/admin/users/{victim.user_id}/api-keys/{key['id']}")).status_code == 204
    assert (await admin.delete(f"/api/admin/users/{victim.user_id}/api-keys/{key['id']}")).status_code == 404
    assert (await admin.get(f"/api/admin/users/{victim.user_id}/api-keys")).json()["keys"][0]["revokedAt"] is not None


async def test_unknown_user_is_404(admin_emails, new_user):
    admin = await new_user(ADMIN_EMAIL)
    missing = "00000000-0000-0000-0000-000000000000"
    assert (await admin.patch(f"/api/admin/users/{missing}", json={"role": "admin"})).status_code == 404

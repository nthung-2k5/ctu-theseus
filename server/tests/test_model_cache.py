import asyncio

import pytest

from theseus.services.model_cache import ModelCache


class _FakeModel:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def cache(monkeypatch):
    """A `ModelCache` whose `_load_sync` is a fast, trackable fake instead
    of a real S3 download + a backend's own load() — these tests are about
    the cache's own concurrency/eviction behavior, not any specific backend."""
    load_calls: list[str] = []

    def fake_load_sync(run_id: str, backend_id: str):
        load_calls.append(run_id)
        return _FakeModel(run_id)

    monkeypatch.setattr(ModelCache, "_load_sync", staticmethod(fake_load_sync))
    monkeypatch.setattr("theseus.services.model_cache.resolve_backend", _fake_resolve_backend)
    monkeypatch.setattr("theseus.services.model_cache.cleanup_temp", lambda *a, **k: None)

    instance = ModelCache(max_size=2)
    instance.load_calls = load_calls
    return instance


async def _fake_resolve_backend(run_id: str) -> str:
    return "fake"


async def test_concurrent_requests_for_same_run_load_once(cache):
    """N callers racing for one run should trigger exactly one
    `load()` — the per-run lock, not N duplicate loads."""
    results = await asyncio.gather(*[cache.get("run-1") for _ in range(10)])

    assert cache.load_calls == ["run-1"]
    assert all(r.run_id == "run-1" for r in results)


async def test_lru_evicts_the_least_recently_used_run(cache):
    await cache.get("run-1")
    await cache.get("run-2")
    await cache.get("run-3")  # cache size 2 — evicts run-1

    assert set(cache._models.keys()) == {"run-2", "run-3"}


async def test_eviction_closes_the_evicted_model(cache):
    run_1 = await cache.get("run-1")
    await cache.get("run-2")
    await cache.get("run-3")  # cache size 2 — evicts run-1

    assert run_1.closed is True


async def test_touching_a_cached_run_protects_it_from_eviction(cache):
    await cache.get("run-1")
    await cache.get("run-2")
    await cache.get("run-1")  # run-2 is now the least recently used
    await cache.get("run-3")

    assert set(cache._models.keys()) == {"run-1", "run-3"}


async def test_failed_load_does_not_poison_the_cache(monkeypatch):
    attempts = {"count": 0}

    def flaky_load(run_id: str, backend_id: str):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated load failure")
        return _FakeModel(run_id)

    monkeypatch.setattr(ModelCache, "_load_sync", staticmethod(flaky_load))
    monkeypatch.setattr("theseus.services.model_cache.resolve_backend", _fake_resolve_backend)
    monkeypatch.setattr("theseus.services.model_cache.cleanup_temp", lambda *a, **k: None)
    cache = ModelCache(max_size=2)

    with pytest.raises(RuntimeError):
        await cache.get("run-1")
    assert "run-1" not in cache._models

    model = await cache.get("run-1")
    assert model.run_id == "run-1"
    assert attempts["count"] == 2


async def test_warm_populates_the_cache_without_returning_a_value(cache):
    result = await cache.warm("run-1")

    assert result is None
    assert "run-1" in cache._models

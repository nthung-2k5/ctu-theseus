import asyncio
import gc
import logging
from collections import OrderedDict

import torch
from config import INFERENCE_MODEL_CACHE_SIZE
from ludwig.api import LudwigModel
from services.storage import cleanup_temp, download_model, find_model_dir

logger = logging.getLogger(__name__)

# Bounds how many models may be *loading* (S3 download + LudwigModel.load)
# at once, independent of how many are already cached — a burst of cold
# requests for distinct runs shouldn't all deserialize onto the GPU at once.
_MAX_CONCURRENT_LOADS = 2


class ModelCache:
    """
    In-process LRU cache of loaded `LudwigModel` instances, keyed by run id.

    `LudwigModel.load()` deserializes the full checkpoint, config, and
    `training_set_metadata` — expensive enough that doing it on every
    inference request (the previous behavior) dominated request latency and
    reliably blew past the gateway's request timeout on a cold run. This
    cache keeps up to `max_size` models resident; the least-recently-used
    one is evicted — dropping the reference and clearing its on-disk
    download cache via `cleanup_temp` — once that cap is exceeded.
    """

    def __init__(self, max_size: int = INFERENCE_MODEL_CACHE_SIZE):
        self._max_size = max_size
        self._models: OrderedDict[str, LudwigModel] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._load_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LOADS)

    async def _lock_for(self, run_id: str) -> asyncio.Lock:
        """Per-run lock so N concurrent requests for the *same* run await
        one load instead of racing N separate (expensive) load calls that
        would otherwise all land in the cache redundantly. Guarded by
        `_locks_guard` since dict lookup-then-insert isn't atomic across an
        `await`."""
        async with self._locks_guard:
            lock = self._locks.get(run_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[run_id] = lock
            return lock

    async def get(self, run_id: str) -> LudwigModel:
        """Return the cached model for `run_id`, loading it first on a miss."""
        if run_id in self._models:
            self._models.move_to_end(run_id)
            return self._models[run_id]

        lock = await self._lock_for(run_id)
        async with lock:
            # Another coroutine may have loaded it while we waited for the lock.
            if run_id in self._models:
                self._models.move_to_end(run_id)
                return self._models[run_id]

            async with self._load_semaphore:
                # Not cached on failure — the exception propagates and the
                # next call (this run or another) retries from scratch,
                # instead of a broken load sticking around as a false hit.
                model = await asyncio.to_thread(self._load_sync, run_id)

            self._models[run_id] = model
            self._models.move_to_end(run_id)
            await self._evict_excess()
            return model

    async def warm(self, run_id: str) -> None:
        """Preload `run_id` without returning it — used by the `warm`
        subject handler so the first real request doesn't pay the
        cold-start cost. A no-op if already cached."""
        await self.get(run_id)

    @staticmethod
    def _load_sync(run_id: str) -> LudwigModel:
        model_dir = find_model_dir(download_model(run_id))
        return LudwigModel.load(model_dir)

    async def _evict_excess(self) -> None:
        while len(self._models) > self._max_size:
            evicted_run_id, evicted_model = self._models.popitem(last=False)
            del evicted_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            await asyncio.to_thread(cleanup_temp, "models", evicted_run_id)
            logger.info(f"Evicted model for run {evicted_run_id} from inference cache")


# Module-level singleton, mirrors services.nats.nats_service
model_cache = ModelCache()

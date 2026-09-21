"""Dedicated executors, one per lane.

Training must never run in the default asyncio.to_thread pool: a multi-hour thread would squat
in a pool that also serves every S3 call, parquet read and password hash in the process.

  train      1 thread   the GPU is singular
  export     2 threads  preserves the old MAX_CONCURRENCY=2; zlib releases the GIL, so threads suffice
  inference  default    short jobs sharing the process with GPU-resident cached models
"""

import asyncio
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

train_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train")
export_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="export")


async def run_in_executor[T](executor: Executor | None, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run fn in an executor with the caller contextvars.

    loop.run_in_executor does NOT copy the context (asyncio.to_thread does, but always uses the
    default pool). The run-log handler and the OpenTelemetry span both depend on contextvars
    reaching the training thread, so copy it explicitly.
    """
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(executor, call)

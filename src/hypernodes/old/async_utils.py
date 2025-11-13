"""Utilities for managing event loops and awaiting coroutines in different contexts."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, TypeVar

T = TypeVar("T")

_THREAD_LOCAL = threading.local()
_EXECUTOR_LOCK = threading.Lock()
_BACKGROUND_EXECUTOR: ThreadPoolExecutor | None = None


def _get_background_executor() -> ThreadPoolExecutor:
    """Lazily create a shared thread pool for running coroutines."""
    global _BACKGROUND_EXECUTOR
    with _EXECUTOR_LOCK:
        if _BACKGROUND_EXECUTOR is None:
            _BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=4)
        return _BACKGROUND_EXECUTOR


def _run_in_background_thread(coro: Awaitable[T]) -> T:
    """Execute coroutine using asyncio.run() inside a worker thread."""
    executor = _get_background_executor()
    future = executor.submit(asyncio.run, coro)
    return future.result()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Return a thread-local event loop, creating it if necessary."""
    loop = getattr(_THREAD_LOCAL, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _THREAD_LOCAL.loop = loop
    return loop


def run_coroutine_sync(coro: Awaitable[T], strategy: str = "per_call") -> T:
    """Run a coroutine to completion according to the configured strategy.

    Always respects notebook/Jupyter constraints: if a loop is already running
    in the current thread, offload to a background worker thread.

    Args:
        coro: Coroutine to execute.
        strategy: One of {"per_call", "thread_local"}.

    Returns:
        Result of the coroutine.
    """
    # Never attempt to run a loop in-thread if one is already running (Jupyter)
    try:
        asyncio.get_running_loop()
        return _run_in_background_thread(coro)
    except RuntimeError:
        pass

    if strategy == "thread_local":
        loop = _get_thread_loop()
        return loop.run_until_complete(coro)

    # Default: create a fresh event loop for this call
    return asyncio.run(coro)

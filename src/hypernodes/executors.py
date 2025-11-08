"""Executor adapters for uniform concurrent.futures-like interface.

This module provides adapters that make sequential and async execution
follow the same submit() interface as concurrent.futures executors.

Key principle: Everything uses the same interface:
    future = executor.submit(fn, *args, **kwargs)
    result = future.result()

This works for:
- SequentialExecutor (custom adapter)
- AsyncExecutor (custom adapter)
- ThreadPoolExecutor (stdlib)
- ProcessPoolExecutor (stdlib)
"""

import asyncio
import inspect
import os
import threading
from concurrent.futures import Future
from typing import Any, Callable


# Default worker counts for different execution strategies
DEFAULT_WORKERS = {
    "sequential": 1,
    "async": 100,  # High concurrency for I/O-bound work
    "threaded": os.cpu_count() or 4,
    "parallel": os.cpu_count() or 4,
}


class SequentialExecutor:
    """Adapter that provides submit() interface for immediate synchronous execution.

    This executor runs functions immediately and synchronously when submit() is called.
    It provides a concurrent.futures-compatible interface but doesn't actually
    use any concurrency.

    Useful for:
    - Debugging pipelines with step-by-step execution
    - Avoiding threading/process overhead for simple pipelines
    - Testing without concurrency complications
    """

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """Execute function immediately and return completed Future.

        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that is already completed with result or exception
        """
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor (no-op for sequential).

        Args:
            wait: Unused (kept for interface compatibility)
        """
        pass


class AsyncExecutor:
    """Adapter that provides submit() interface for asyncio-based concurrent execution.

    This executor runs async functions concurrently using asyncio in a background thread.
    It can also handle sync functions by running them in the executor.

    Useful for:
    - I/O-bound workloads (API calls, file I/O, database queries)
    - High-concurrency scenarios (100+ concurrent tasks)
    - Mixed async/sync functions in the same pipeline

    Note: This executor runs an event loop in a background thread, allowing
    async tasks to execute concurrently while providing a sync interface.
    """

    def __init__(self, max_workers: int = DEFAULT_WORKERS["async"]):
        """Initialize AsyncExecutor with a background event loop.

        Args:
            max_workers: Maximum concurrent tasks (default 100)
        """
        self.max_workers = max_workers
        self._loop = None
        self._thread = None
        self._semaphore = None
        self._start_loop()

    def _start_loop(self):
        """Start event loop in background thread."""
        self._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Initialize semaphore in the loop
        future = asyncio.run_coroutine_threadsafe(
            self._init_semaphore(), self._loop
        )
        future.result()  # Wait for initialization

    async def _init_semaphore(self):
        """Initialize semaphore in the event loop."""
        self._semaphore = asyncio.Semaphore(self.max_workers)

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """Execute function concurrently using asyncio.

        Handles both async and sync functions. Async functions run directly,
        sync functions run in executor pool.

        Args:
            fn: Function to execute (can be async or sync)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that will be completed when function finishes
        """
        async def run_with_semaphore():
            """Run function with concurrency control."""
            async with self._semaphore:
                if asyncio.iscoroutinefunction(fn):
                    # Async function - await directly
                    return await fn(*args, **kwargs)
                else:
                    # Sync function - run in executor
                    return await self._loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        # Schedule coroutine in background loop
        future = asyncio.run_coroutine_threadsafe(run_with_semaphore(), self._loop)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor and cleanup event loop.

        Args:
            wait: If True, wait for all pending tasks to complete
        """
        if self._loop is not None:
            if wait:
                # Give pending tasks time to complete
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    gather_future = asyncio.gather(*pending, return_exceptions=True)
                    asyncio.run_coroutine_threadsafe(gather_future, self._loop).result(timeout=30)

            # Stop the loop
            self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)

            self._loop = None
            self._thread = None
            self._semaphore = None

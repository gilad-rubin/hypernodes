"""Tests for executor adapters - Phase 1 of SOLID refactoring.

These tests verify that SequentialExecutor and AsyncExecutor provide
a uniform interface compatible with concurrent.futures.Executor.
"""

import asyncio
import pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from hypernodes.executors import (
    SequentialExecutor,
    AsyncExecutor,
    DEFAULT_WORKERS,
)


class TestSequentialExecutor:
    """Test SequentialExecutor provides concurrent.futures-like interface."""

    def test_sequential_executor_submit(self):
        """Test SequentialExecutor executes immediately."""
        executor = SequentialExecutor()
        future = executor.submit(lambda x: x * 2, 5)
        assert future.result() == 10

    def test_sequential_executor_exception(self):
        """Test SequentialExecutor handles exceptions."""
        executor = SequentialExecutor()
        future = executor.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            future.result()

    def test_sequential_executor_with_kwargs(self):
        """Test SequentialExecutor handles keyword arguments."""
        executor = SequentialExecutor()
        future = executor.submit(lambda x, y: x + y, 3, y=4)
        assert future.result() == 7

    def test_sequential_executor_shutdown(self):
        """Test SequentialExecutor has shutdown method."""
        executor = SequentialExecutor()
        executor.shutdown(wait=True)  # Should not raise


class TestAsyncExecutor:
    """Test AsyncExecutor provides concurrent.futures-like interface."""

    def test_async_executor_submit(self):
        """Test AsyncExecutor runs async functions."""
        executor = AsyncExecutor()

        async def async_fn(x):
            await asyncio.sleep(0.01)
            return x * 2

        future = executor.submit(async_fn, 5)
        assert future.result() == 10

    def test_async_executor_sync_function(self):
        """Test AsyncExecutor can also run sync functions."""
        executor = AsyncExecutor()

        def sync_fn(x):
            return x * 2

        future = executor.submit(sync_fn, 5)
        assert future.result() == 10

    def test_async_executor_exception(self):
        """Test AsyncExecutor handles exceptions."""
        executor = AsyncExecutor()

        async def failing_fn():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        future = executor.submit(failing_fn)
        with pytest.raises(ValueError, match="test error"):
            future.result()

    def test_async_executor_concurrency(self):
        """Test AsyncExecutor runs tasks concurrently."""
        import time

        executor = AsyncExecutor()

        async def slow_fn(x):
            await asyncio.sleep(0.1)
            return x

        # Submit multiple tasks
        start = time.time()
        futures = [executor.submit(slow_fn, i) for i in range(5)]
        results = [f.result() for f in futures]
        duration = time.time() - start

        assert results == [0, 1, 2, 3, 4]
        # Should complete in ~0.1s (concurrent), not 0.5s (sequential)
        assert duration < 0.2

        executor.shutdown(wait=True)

    def test_async_executor_shutdown(self):
        """Test AsyncExecutor has shutdown method."""
        executor = AsyncExecutor()
        executor.shutdown(wait=True)  # Should not raise


def _test_function(x):
    """Helper function for testing (picklable for ProcessPoolExecutor)."""
    return x * 2


class TestExecutorInterfaceCompatibility:
    """Test all executors have compatible interface."""

    def test_executor_interface_compatibility(self):
        """Test all executors have compatible interface."""
        executors = [
            SequentialExecutor(),
            AsyncExecutor(),
            ThreadPoolExecutor(max_workers=2),
            ProcessPoolExecutor(max_workers=2),
        ]

        for executor in executors:
            assert hasattr(executor, "submit")
            assert hasattr(executor, "shutdown")
            future = executor.submit(_test_function, 5)
            assert future.result() == 10

        # Cleanup
        for ex in executors[1:]:  # Cleanup all except Sequential (which doesn't need it)
            ex.shutdown(wait=True)

    def test_default_workers_configuration(self):
        """Test DEFAULT_WORKERS has expected structure."""
        assert "sequential" in DEFAULT_WORKERS
        assert "async" in DEFAULT_WORKERS
        assert "threaded" in DEFAULT_WORKERS
        assert "parallel" in DEFAULT_WORKERS

        assert DEFAULT_WORKERS["sequential"] == 1
        assert DEFAULT_WORKERS["async"] >= 100
        assert DEFAULT_WORKERS["threaded"] >= 1
        assert DEFAULT_WORKERS["parallel"] >= 1

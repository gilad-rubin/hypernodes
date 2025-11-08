"""Tests for AsyncExecutor auto-wrapping of sync functions.

These tests verify that AsyncExecutor can automatically wrap synchronous
blocking functions to run in thread pools, providing concurrency benefits
without requiring async/await syntax.
"""

import asyncio
import time

import pytest

from hypernodes import HypernodesEngine, Pipeline, node
from hypernodes.executors import AsyncExecutor


class TestAsyncExecutorAutoWrapping:
    """Test AsyncExecutor auto-wrapping functionality."""

    def test_sync_blocking_function_autowrap(self):
        """Test that sync blocking functions are auto-wrapped and run concurrently."""
        
        @node(output_name="result")
        def blocking_fn(x: int) -> int:
            time.sleep(0.1)  # Blocking I/O
            return x ** 2
        
        pipeline = Pipeline(
            nodes=[blocking_fn],
            backend=HypernodesEngine(map_executor=AsyncExecutor())
        )
        
        # Run 5 items - should complete in ~0.1s (concurrent), not 0.5s (sequential)
        start = time.time()
        results = pipeline.map(inputs={"x": [0, 1, 2, 3, 4]}, map_over="x")
        duration = time.time() - start
        
        assert results["result"] == [0, 1, 4, 9, 16]
        assert duration < 0.3, f"Expected <0.3s, got {duration:.3f}s - not concurrent!"
        
    def test_native_async_function_execution(self):
        """Test that native async functions still work properly."""
        
        @node(output_name="result")
        async def async_fn(x: int) -> int:
            await asyncio.sleep(0.1)
            return x ** 2
        
        pipeline = Pipeline(
            nodes=[async_fn],
            backend=HypernodesEngine(map_executor=AsyncExecutor())
        )
        
        # Run 5 items - should complete in ~0.1s (concurrent)
        start = time.time()
        results = pipeline.map(inputs={"x": [0, 1, 2, 3, 4]}, map_over="x")
        duration = time.time() - start
        
        assert results["result"] == [0, 1, 4, 9, 16]
        assert duration < 0.3, f"Expected <0.3s, got {duration:.3f}s - not concurrent!"
        
    def test_mixed_sync_async_pipeline(self):
        """Test pipeline with both sync and async nodes."""
        
        @node(output_name="sync_result")
        def sync_node(x: int) -> int:
            time.sleep(0.05)
            return x * 2
        
        @node(output_name="async_result")
        async def async_node(sync_result: int) -> int:
            await asyncio.sleep(0.05)
            return sync_result + 10
        
        @node(output_name="final")
        def final_node(async_result: int) -> int:
            return async_result ** 2
        
        pipeline = Pipeline(
            nodes=[sync_node, async_node, final_node],
            backend=HypernodesEngine(node_executor=AsyncExecutor())
        )
        
        result = pipeline.run(inputs={"x": 5})
        
        # x=5 -> sync_node(5)=10 -> async_node(10)=20 -> final_node(20)=400
        assert result["final"] == 400
        
    def test_sync_function_with_complex_args(self):
        """Test auto-wrapped sync function with various argument types."""
        
        @node(output_name="result")
        def complex_fn(x: int, y: str, z: list = None) -> dict:
            time.sleep(0.01)
            return {"x": x, "y": y, "z": z or []}
        
        pipeline = Pipeline(
            nodes=[complex_fn],
            backend=HypernodesEngine(map_executor=AsyncExecutor())
        )
        
        result = pipeline.run(inputs={"x": 42, "y": "test", "z": [1, 2, 3]})
        
        assert result["result"] == {"x": 42, "y": "test", "z": [1, 2, 3]}
        
    def test_sync_function_exception_handling(self):
        """Test that exceptions in auto-wrapped sync functions are properly propagated."""
        
        @node(output_name="result")
        def failing_fn(x: int) -> int:
            time.sleep(0.01)
            if x == 2:
                raise ValueError(f"Test error for x={x}")
            return x ** 2
        
        pipeline = Pipeline(
            nodes=[failing_fn],
            backend=HypernodesEngine(map_executor=AsyncExecutor())
        )
        
        # Should raise ValueError when processing x=2
        with pytest.raises(ValueError, match="Test error for x=2"):
            pipeline.map(inputs={"x": [0, 1, 2, 3]}, map_over="x")
            
    def test_node_level_concurrency_sync_functions(self):
        """Test node-level concurrency with multiple independent sync nodes."""
        
        @node(output_name="result1")
        def task1(x: int) -> int:
            time.sleep(0.1)
            return x * 2
        
        @node(output_name="result2")
        def task2(x: int) -> int:
            time.sleep(0.1)
            return x * 3
        
        @node(output_name="result3")
        def task3(x: int) -> int:
            time.sleep(0.1)
            return x * 4
        
        @node(output_name="final")
        def combine(result1: int, result2: int, result3: int) -> dict:
            return {"r1": result1, "r2": result2, "r3": result3}
        
        pipeline = Pipeline(
            nodes=[task1, task2, task3, combine],
            backend=HypernodesEngine(node_executor=AsyncExecutor())
        )
        
        # 3 independent tasks should run concurrently in ~0.1s, not 0.3s
        start = time.time()
        result = pipeline.run(inputs={"x": 10})
        duration = time.time() - start
        
        assert result["final"] == {"r1": 20, "r2": 30, "r3": 40}
        assert duration < 0.25, f"Expected <0.25s, got {duration:.3f}s - not concurrent!"
        
    def test_high_concurrency_sync_functions(self):
        """Test that many sync functions can run concurrently."""
        
        @node(output_name="result")
        def quick_fn(x: int) -> int:
            time.sleep(0.05)
            return x ** 2
        
        pipeline = Pipeline(
            nodes=[quick_fn],
            backend=HypernodesEngine(map_executor=AsyncExecutor(max_workers=50))
        )
        
        # Run 20 items - should complete in ~0.05s if concurrent
        start = time.time()
        results = pipeline.map(inputs={"x": list(range(20))}, map_over="x")
        duration = time.time() - start
        
        assert len(results["result"]) == 20
        # With high concurrency, should finish in ~0.05-0.1s, not 1.0s
        assert duration < 0.3, f"Expected <0.3s, got {duration:.3f}s - not concurrent!"
        
    def test_jupyter_compatibility(self):
        """Test that AsyncExecutor detects Jupyter environment correctly."""
        executor = AsyncExecutor()
        
        # Should have _in_jupyter attribute
        assert hasattr(executor, "_in_jupyter")
        assert isinstance(executor._in_jupyter, bool)
        
        # Should work regardless of Jupyter status
        @node(output_name="result")
        def test_fn(x: int) -> int:
            time.sleep(0.01)
            return x * 2
        
        pipeline = Pipeline(
            nodes=[test_fn],
            backend=HypernodesEngine(map_executor=executor)
        )
        
        result = pipeline.run(inputs={"x": 5})
        assert result["result"] == 10

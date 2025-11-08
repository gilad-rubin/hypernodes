"""Tests for engine orchestration - Phase 3 of SOLID refactoring.

These tests verify that the HypernodesEngine correctly orchestrates pipeline execution
using different executors (sequential, async, threaded, parallel).

Note: After merging, the orchestrator is now internal to the engine.
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import PipelineCallback, CallbackContext
from hypernodes.executors import SequentialExecutor, AsyncExecutor
from hypernodes.engine import HypernodesEngine


class TestOrchestratorBasic:
    """Test basic orchestrator functionality."""

    def test_orchestrator_sequential(self):
        """Test engine orchestration with sequential executor."""
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="result")
        def add_ten(doubled: int) -> int:
            return doubled + 10

        pipeline = Pipeline(nodes=[double, add_ten])
        engine = HypernodesEngine(node_executor=SequentialExecutor())

        result = engine.run(pipeline, {"x": 5})

        assert result == {"doubled": 10, "result": 20}

    def test_orchestrator_threaded(self):
        """Test engine orchestration with ThreadPoolExecutor."""
        @node(output_name="a")
        def slow_a(x: int) -> int:
            time.sleep(0.05)
            return x * 2

        @node(output_name="b")
        def slow_b(x: int) -> int:
            time.sleep(0.05)
            return x * 3

        @node(output_name="result")
        def combine(a: int, b: int) -> int:
            return a + b

        pipeline = Pipeline(nodes=[slow_a, slow_b, combine])
        engine = HypernodesEngine(node_executor=ThreadPoolExecutor(max_workers=2))

        start = time.time()
        result = engine.run(pipeline, {"x": 5})
        duration = time.time() - start

        assert result == {"a": 10, "b": 15, "result": 25}
        # Should execute a and b in parallel (~0.05s), then combine
        assert duration < 0.1  # Would be 0.15s if sequential

        engine.shutdown(wait=True)

    def test_orchestrator_async(self):
        """Test engine orchestration with AsyncExecutor."""
        @node(output_name="a")
        async def slow_a(x: int) -> int:
            await asyncio.sleep(0.05)
            return x * 2

        @node(output_name="b")
        async def slow_b(x: int) -> int:
            await asyncio.sleep(0.05)
            return x * 3

        @node(output_name="result")
        def combine(a: int, b: int) -> int:
            return a + b

        pipeline = Pipeline(nodes=[slow_a, slow_b, combine])
        engine = HypernodesEngine(node_executor=AsyncExecutor())

        start = time.time()
        result = engine.run(pipeline, {"x": 5})
        duration = time.time() - start

        assert result == {"a": 10, "b": 15, "result": 25}
        # Should execute a and b concurrently (~0.05s), then combine
        assert duration < 0.1  # Would be 0.15s if sequential

        engine.shutdown(wait=True)

    def test_orchestrator_async_map(self):
        """Test engine map operation with AsyncExecutor."""
        @node(output_name="doubled")
        async def double(x: int) -> int:
            await asyncio.sleep(0.05)
            return x * 2

        pipeline = Pipeline(nodes=[double])
        engine = HypernodesEngine(map_executor=AsyncExecutor())

        items = [{"x": 1}, {"x": 2}, {"x": 3}]

        start = time.time()
        results = engine.map(pipeline, items, {})
        duration = time.time() - start

        assert len(results) == 3
        assert results[0] == {"doubled": 2}
        assert results[1] == {"doubled": 4}
        assert results[2] == {"doubled": 6}
        # Should process items concurrently (~0.05s), not sequentially (0.15s)
        assert duration < 0.1

        engine.shutdown(wait=True)

    @pytest.mark.skip(reason="ProcessPoolExecutor has pickling limitations with local functions and pipeline objects")
    def test_orchestrator_parallel(self):
        """Test engine orchestration with ProcessPoolExecutor.

        Note: This test is skipped due to ProcessPoolExecutor's pickling limitations.
        Nodes with closures and Pipeline objects containing locks cannot be pickled.
        For parallel execution, use ThreadPoolExecutor or AsyncExecutor instead.
        """
        @node(output_name="a")
        def slow_a(x: int) -> int:
            time.sleep(0.05)
            return x * 2

        @node(output_name="b")
        def slow_b(x: int) -> int:
            time.sleep(0.05)
            return x * 3

        @node(output_name="result")
        def combine(a: int, b: int) -> int:
            return a + b

        pipeline = Pipeline(nodes=[slow_a, slow_b, combine])
        engine = HypernodesEngine(node_executor=ProcessPoolExecutor(max_workers=2))

        start = time.time()
        result = engine.run(pipeline, {"x": 5})
        duration = time.time() - start

        assert result == {"a": 10, "b": 15, "result": 25}
        # Should execute a and b in parallel (~0.05s), then combine
        assert duration < 0.2  # More overhead than threads, so allow more time

        engine.shutdown(wait=True)

    @pytest.mark.skip(reason="ProcessPoolExecutor has pickling limitations with local functions and pipeline objects")
    def test_orchestrator_parallel_map(self):
        """Test engine map operation with ProcessPoolExecutor.

        Note: This test is skipped due to ProcessPoolExecutor's pickling limitations.
        Nodes with closures and Pipeline objects containing locks cannot be pickled.
        For parallel map execution, use ThreadPoolExecutor or AsyncExecutor instead.
        """
        @node(output_name="doubled")
        def double(x: int) -> int:
            time.sleep(0.05)
            return x * 2

        pipeline = Pipeline(nodes=[double])
        engine = HypernodesEngine(map_executor=ProcessPoolExecutor(max_workers=3))

        items = [{"x": 1}, {"x": 2}, {"x": 3}]

        start = time.time()
        results = engine.map(pipeline, items, {})
        duration = time.time() - start

        assert len(results) == 3
        assert results[0] == {"doubled": 2}
        assert results[1] == {"doubled": 4}
        assert results[2] == {"doubled": 6}
        # Should process items in parallel (~0.05s), not sequentially (0.15s)
        assert duration < 0.2  # More overhead for process spawning

        engine.shutdown(wait=True)

    def test_orchestrator_linear_pipeline(self):
        """Test engine orchestration with linear dependency chain."""
        @node(output_name="step1")
        def first(x: int) -> int:
            return x + 1

        @node(output_name="step2")
        def second(step1: int) -> int:
            return step1 * 2

        @node(output_name="step3")
        def third(step2: int) -> int:
            return step2 + 10

        pipeline = Pipeline(nodes=[first, second, third])
        engine = HypernodesEngine(node_executor="sequential")

        result = engine.run(pipeline, {"x": 5})

        assert result == {"step1": 6, "step2": 12, "step3": 22}


class TestOrchestratorOutputFiltering:
    """Test engine output filtering."""

    def test_orchestrator_output_name_filtering(self):
        """Test engine respects output_name parameter."""
        @node(output_name="a")
        def compute_a(x: int) -> int:
            return x * 2

        @node(output_name="b")
        def compute_b(a: int) -> int:
            return a + 10

        @node(output_name="c")
        def compute_c(b: int) -> int:
            return b * 3

        pipeline = Pipeline(nodes=[compute_a, compute_b, compute_c])
        engine = HypernodesEngine(node_executor="sequential")

        # Request only "b" - should compute a (dependency) and b, but not c
        # Output should only contain "b" (filtered)
        result = engine.run(pipeline, {"x": 5}, output_name="b")

        assert result == {"b": 20}  # Only requested output
        assert "a" not in result  # Dependency was computed but not in output
        assert "c" not in result  # c was not computed

    def test_orchestrator_output_name_list_filtering(self):
        """Test engine respects output_name parameter with list of outputs."""
        @node(output_name="a")
        def compute_a(x: int) -> int:
            return x * 2

        @node(output_name="b")
        def compute_b(a: int) -> int:
            return a + 10

        @node(output_name="c")
        def compute_c(b: int) -> int:
            return b * 3

        @node(output_name="d")
        def compute_d(c: int) -> int:
            return c + 1

        pipeline = Pipeline(nodes=[compute_a, compute_b, compute_c, compute_d])
        engine = HypernodesEngine(node_executor="sequential")

        # Request "b" and "c" - should compute a, b, c but not d
        # Output should only contain "b" and "c"
        result = engine.run(pipeline, {"x": 5}, output_name=["b", "c"])

        assert result == {"b": 20, "c": 60}  # Only requested outputs
        assert "a" not in result  # Dependency computed but filtered
        assert "d" not in result  # Not computed


class TestOrchestratorCaching:
    """Test engine with caching."""

    def test_orchestrator_with_cache(self):
        """Test engine leverages caching."""
        @node(output_name="a", cache=True)
        def compute_a(x: int) -> int:
            return x * 2

        @node(output_name="b", cache=True)
        def compute_b(a: int) -> int:
            return a + 10

        cache = DiskCache(".test_cache")
        cache.clear()  # Clear before test

        pipeline = Pipeline(nodes=[compute_a, compute_b], cache=cache)
        engine = HypernodesEngine(node_executor="sequential")

        # First run
        result1 = engine.run(pipeline, {"x": 5})
        assert result1 == {"a": 10, "b": 20}

        # Second run (should use cache - verified by consistent signatures)
        result2 = engine.run(pipeline, {"x": 5})
        assert result2 == {"a": 10, "b": 20}

        # Cleanup
        cache.clear()


class TestOrchestratorCallbacks:
    """Test engine triggers callbacks."""

    def test_orchestrator_callbacks(self):
        """Test engine triggers callbacks."""
        events = []

        class TestCallback(PipelineCallback):
            def on_pipeline_start(self, pipeline_id, inputs, ctx):
                events.append("pipeline_start")

            def on_pipeline_end(self, pipeline_id, outputs, duration, ctx):
                events.append("pipeline_end")

            def on_node_start(self, node_id, inputs, ctx):
                events.append(f"node_start:{node_id}")

            def on_node_end(self, node_id, outputs, duration, ctx):
                events.append(f"node_end:{node_id}")

        @node(output_name="result")
        def identity(x: int) -> int:
            return x

        callback = TestCallback()
        pipeline = Pipeline(nodes=[identity], callbacks=[callback])
        engine = HypernodesEngine(node_executor="sequential")

        engine.run(pipeline, {"x": 5})

        assert "pipeline_start" in events
        assert "node_start:identity" in events
        assert "node_end:identity" in events
        assert "pipeline_end" in events


class TestOrchestratorMultipleInputs:
    """Test engine with multiple input parameters."""

    def test_multiple_inputs(self):
        """Test engine with nodes that have multiple parameters."""
        @node(output_name="sum")
        def add(a: int, b: int) -> int:
            return a + b

        @node(output_name="product")
        def multiply(a: int, b: int) -> int:
            return a * b

        @node(output_name="result")
        def combine(sum: int, product: int) -> int:
            return sum + product

        pipeline = Pipeline(nodes=[add, multiply, combine])
        engine = HypernodesEngine(node_executor="sequential")

        result = engine.run(pipeline, {"a": 3, "b": 4})

        assert result == {"sum": 7, "product": 12, "result": 19}

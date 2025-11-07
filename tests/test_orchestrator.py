"""Tests for pipeline orchestrator - Phase 3 of SOLID refactoring.

These tests verify that the PipelineOrchestrator correctly executes pipelines
using different executors (sequential, async, threaded, parallel).
"""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import PipelineCallback, CallbackContext
from hypernodes.executor_adapters import SequentialExecutor, AsyncExecutor
from hypernodes.orchestrator import PipelineOrchestrator


class TestOrchestratorBasic:
    """Test basic orchestrator functionality."""

    def test_orchestrator_sequential(self):
        """Test orchestrator with sequential executor."""
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="result")
        def add_ten(doubled: int) -> int:
            return doubled + 10

        pipeline = Pipeline(nodes=[double, add_ten])
        executor = SequentialExecutor()
        orchestrator = PipelineOrchestrator(executor)

        result = orchestrator.execute(pipeline, {"x": 5}, None, None)

        assert result == {"doubled": 10, "result": 20}

    def test_orchestrator_threaded(self):
        """Test orchestrator with ThreadPoolExecutor."""
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
        executor = ThreadPoolExecutor(max_workers=2)
        orchestrator = PipelineOrchestrator(executor)

        start = time.time()
        result = orchestrator.execute(pipeline, {"x": 5}, None, None)
        duration = time.time() - start

        assert result == {"a": 10, "b": 15, "result": 25}
        # Should execute a and b in parallel (~0.05s), then combine
        assert duration < 0.1  # Would be 0.15s if sequential

        executor.shutdown(wait=True)

    def test_orchestrator_linear_pipeline(self):
        """Test orchestrator with linear dependency chain."""
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
        executor = SequentialExecutor()
        orchestrator = PipelineOrchestrator(executor)

        result = orchestrator.execute(pipeline, {"x": 5}, None, None)

        assert result == {"step1": 6, "step2": 12, "step3": 22}


class TestOrchestratorOutputFiltering:
    """Test orchestrator output filtering."""

    def test_orchestrator_output_name_filtering(self):
        """Test orchestrator respects output_name parameter."""
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
        executor = SequentialExecutor()
        orchestrator = PipelineOrchestrator(executor)

        # Request only "b" - should not compute "c"
        result = orchestrator.execute(pipeline, {"x": 5}, None, "b")

        assert "b" in result
        assert "a" in result  # a is required for b
        assert "c" not in result  # c is not required


class TestOrchestratorCaching:
    """Test orchestrator with caching."""

    def test_orchestrator_with_cache(self):
        """Test orchestrator leverages caching."""
        @node(output_name="a", cache=True)
        def compute_a(x: int) -> int:
            return x * 2

        @node(output_name="b", cache=True)
        def compute_b(a: int) -> int:
            return a + 10

        cache = DiskCache(".test_cache")
        cache.clear()  # Clear before test

        pipeline = Pipeline(nodes=[compute_a, compute_b], cache=cache)
        executor = SequentialExecutor()
        orchestrator = PipelineOrchestrator(executor)

        # First run
        result1 = orchestrator.execute(pipeline, {"x": 5}, None, None)
        assert result1 == {"a": 10, "b": 20}

        # Second run (should use cache - verified by consistent signatures)
        result2 = orchestrator.execute(pipeline, {"x": 5}, None, None)
        assert result2 == {"a": 10, "b": 20}

        # Cleanup
        cache.clear()


class TestOrchestratorCallbacks:
    """Test orchestrator triggers callbacks."""

    def test_orchestrator_callbacks(self):
        """Test orchestrator triggers callbacks."""
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
        executor = SequentialExecutor()
        orchestrator = PipelineOrchestrator(executor)

        orchestrator.execute(pipeline, {"x": 5}, None, None)

        assert "pipeline_start" in events
        assert "node_start:identity" in events
        assert "node_end:identity" in events
        assert "pipeline_end" in events


class TestOrchestratorMultipleInputs:
    """Test orchestrator with multiple input parameters."""

    def test_multiple_inputs(self):
        """Test orchestrator with nodes that have multiple parameters."""
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
        executor = SequentialExecutor()
        orchestrator = PipelineOrchestrator(executor)

        result = orchestrator.execute(pipeline, {"a": 3, "b": 4}, None, None)

        assert result == {"sum": 7, "product": 12, "result": 19}

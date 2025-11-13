"""Tests for execution engine - Phase 4 of SOLID refactoring.

These tests verify that the HypernodesEngine correctly:
- Resolves executor specifications (strings → instances)
- Handles basic pipeline execution
- Manages map operations
- Coordinates node execution via executors
"""

import pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from hypernodes import node, Pipeline
from hypernodes.engine import HypernodesEngine, Engine
from hypernodes.executors import SequentialExecutor, AsyncExecutor
# CloudpickleProcessPoolExecutor replaced with loky


class TestEngineBasic:
    """Test basic engine functionality."""

    def test_engine_basic_run(self):
        """Test basic pipeline execution."""
        @node(output_name="result")
        def double(x: int) -> int:
            return x * 2

        pipeline = Pipeline(nodes=[double])
        engine = HypernodesEngine()

        # Call engine directly (with_engine integration will be added during migration)
        result = engine.run(pipeline, {"x": 5})
        assert result == {"result": 10}

    def test_engine_is_abstract(self):
        """Test that Engine is an abstract base class."""
        assert hasattr(Engine, '__abstractmethods__')
        with pytest.raises(TypeError):
            Engine()  # Cannot instantiate abstract class


class TestEngineExecutorResolution:
    """Test executor resolution from strings."""

    def test_engine_string_executor_sequential(self):
        """Test engine resolves 'sequential' executor."""
        engine = HypernodesEngine(node_executor="sequential")
        assert isinstance(engine.node_executor, SequentialExecutor)

    def test_engine_string_executor_threaded(self):
        """Test engine resolves 'threaded' executor."""
        engine = HypernodesEngine(node_executor="threaded")
        assert isinstance(engine.node_executor, ThreadPoolExecutor)

    def test_engine_string_executor_parallel_disabled_for_node(self):
        """Node-level 'parallel' is disabled; users must use 'threaded' or map_executor='parallel'."""
        with pytest.raises(ValueError):
            HypernodesEngine(node_executor="parallel")

    def test_engine_string_executor_async(self):
        """Test engine resolves 'async' executor."""
        engine = HypernodesEngine(node_executor="async")
        assert isinstance(engine.node_executor, AsyncExecutor)
        engine.node_executor.shutdown(wait=True)

    def test_engine_custom_executor(self):
        """Test engine accepts custom executor instances."""
        custom_executor = ThreadPoolExecutor(max_workers=8)
        engine = HypernodesEngine(node_executor=custom_executor)

        assert engine.node_executor is custom_executor

        custom_executor.shutdown(wait=True)

    def test_engine_default_executors(self):
        """Test engine uses sensible defaults."""
        engine = HypernodesEngine()

        # Should default to sequential for both
        assert isinstance(engine.node_executor, SequentialExecutor)
        assert isinstance(engine.map_executor, SequentialExecutor)


class TestEngineMapOperations:
    """Test engine map operations."""

    def test_engine_map_basic(self):
        """Test basic map operation."""
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        pipeline = Pipeline(nodes=[double])
        engine = HypernodesEngine()

        # Map interface - items are passed as a list
        items = [{"x": 1}, {"x": 2}, {"x": 3}]
        results = engine.map(pipeline, items, {}, None)

        assert len(results) == 3
        assert results[0] == {"doubled": 2}
        assert results[1] == {"doubled": 4}
        assert results[2] == {"doubled": 6}


class TestEngineDifferentExecutors:
    """Test using different executors for nodes vs maps."""

    def test_engine_different_node_and_map_executors(self):
        """Test using different executors for nodes vs maps."""
        @node(output_name="result")
        def double(x: int) -> int:
            return x * 2

        pipeline = Pipeline(nodes=[double])

        # Sequential for nodes, threaded for maps
        engine = HypernodesEngine(
            node_executor="sequential",
            map_executor=ThreadPoolExecutor(max_workers=2)
        )

        # Regular run uses node_executor
        result = engine.run(pipeline, {"x": 5})
        assert result == {"result": 10}

        # Map uses map_executor
        items = [{"x": 1}, {"x": 2}]
        results = engine.map(pipeline, items, {}, None)
        assert len(results) == 2

        engine.map_executor.shutdown(wait=True)


class TestEngineComplexPipelines:
    """Test engine with complex pipelines."""

    def test_engine_diamond_dependency(self):
        """Test engine with diamond dependency pattern."""
        @node(output_name="a")
        def compute_a(x: int) -> int:
            return x * 2

        @node(output_name="b")
        def compute_b(a: int) -> int:
            return a + 1

        @node(output_name="c")
        def compute_c(a: int) -> int:
            return a * 3

        @node(output_name="result")
        def combine(b: int, c: int) -> int:
            return b + c

        pipeline = Pipeline(nodes=[compute_a, compute_b, compute_c, combine])
        engine = HypernodesEngine(node_executor="threaded")

        result = engine.run(pipeline, {"x": 5})

        # x=5 → a=10 → b=11, c=30 → result=41
        assert result == {"a": 10, "b": 11, "c": 30, "result": 41}

    def test_engine_with_output_name_filtering(self):
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
        engine = HypernodesEngine()

        # Request only "b"
        result = engine.run(pipeline, {"x": 5}, output_name="b")

        # Should compute a and b, but not c
        # Output should only contain "b"
        assert result == {"b": 20}
        assert "a" not in result  # Dependency computed but filtered from output
        assert "c" not in result  # Not computed

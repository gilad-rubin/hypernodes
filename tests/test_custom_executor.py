"""Test custom executor support."""

from concurrent.futures import ThreadPoolExecutor

from hypernodes import HypernodesEngine, Pipeline, node


def test_custom_executor_with_threadpool():
    """Test that we can pass a ThreadPoolExecutor as a custom executor."""

    @node(output_name="result")
    def double(x: int) -> int:
        return x * 2

    # Use standard library ThreadPoolExecutor as custom executor
    custom_executor = ThreadPoolExecutor(max_workers=2)

    try:
        engine = HypernodesEngine(node_executor=custom_executor)
        pipeline = Pipeline(nodes=[double], engine=engine)

        result = pipeline.run(inputs={"x": 5})
        assert result["result"] == 10
    finally:
        custom_executor.shutdown(wait=True)


def test_custom_executor_protocol_implementation():
    """Test that a custom class implementing Executor protocol works."""

    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1

    # Custom executor that logs calls
    class LoggingExecutor:
        """Custom executor that logs all submissions."""

        def __init__(self):
            self.call_count = 0
            self.underlying = ThreadPoolExecutor(max_workers=1)

        def submit(self, fn, *args, **kwargs):
            """Submit work and log the call."""
            self.call_count += 1
            return self.underlying.submit(fn, *args, **kwargs)

        def shutdown(self, wait=True):
            """Cleanup resources."""
            self.underlying.shutdown(wait=wait)

    executor = LoggingExecutor()
    try:
        engine = HypernodesEngine(node_executor=executor)
        pipeline = Pipeline(nodes=[add_one], engine=engine)

        result = pipeline.run(inputs={"x": 10})
        assert result["result"] == 11
        # Verify our custom executor was actually used
        assert executor.call_count > 0
    finally:
        executor.shutdown(wait=True)


def test_mixed_builtin_and_custom_executors():
    """Test using a custom executor for nodes and built-in for map."""

    @node(output_name="result")
    def triple(x: int) -> int:
        return x * 3

    custom_node_executor = ThreadPoolExecutor(max_workers=4)

    try:
        # Custom executor for nodes, built-in for map
        engine = HypernodesEngine(
            node_executor=custom_node_executor, map_executor="sequential"
        )
        pipeline = Pipeline(nodes=[triple], engine=engine)

        # Test single run
        result = pipeline.run(inputs={"x": 3})
        assert result["result"] == 9

        # Test map
        results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results == {"result": [3, 6, 9]}
    finally:
        custom_node_executor.shutdown(wait=True)


def test_type_annotation_accepts_executor_protocol():
    """Test that type hints properly accept Executor protocol."""
    from typing import get_type_hints

    # This test verifies that our type hints are correct
    hints = get_type_hints(HypernodesEngine.__init__)

    # node_executor should accept either strings or Executor
    assert "node_executor" in hints
    assert "map_executor" in hints

    # These should not raise type errors
    engine1 = HypernodesEngine(node_executor="sequential")
    engine2 = HypernodesEngine(node_executor=ThreadPoolExecutor(max_workers=2))

    engine2.shutdown()

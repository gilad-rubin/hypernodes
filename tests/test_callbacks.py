"""Tests for callback behavior with SeqEngine."""

import tempfile

from hypernodes import DiskCache, Pipeline, PipelineCallback, SeqEngine, node
from hypernodes.callbacks import CallbackContext


class TrackingCallback(PipelineCallback):
    """Callback that tracks all events."""

    def __init__(self):
        self.events = []

    def on_pipeline_start(
        self, pipeline_id: str, inputs: dict, ctx: CallbackContext
    ) -> None:
        self.events.append(("pipeline_start", pipeline_id))

    def on_pipeline_end(
        self, pipeline_id: str, outputs: dict, duration: float, ctx: CallbackContext
    ) -> None:
        self.events.append(("pipeline_end", pipeline_id))

    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext) -> None:
        self.events.append(("node_start", node_id))

    def on_node_end(
        self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext
    ) -> None:
        self.events.append(("node_end", node_id))

    def on_node_cached(
        self, node_id: str, signature: str, ctx: CallbackContext
    ) -> None:
        self.events.append(("node_cached", node_id))

    def on_nested_pipeline_start(
        self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext
    ) -> None:
        self.events.append(("nested_start", parent_id, child_pipeline_id))

    def on_nested_pipeline_end(
        self,
        parent_id: str,
        child_pipeline_id: str,
        duration: float,
        ctx: CallbackContext,
    ) -> None:
        self.events.append(("nested_end", parent_id, child_pipeline_id))

    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        self.events.append(("map_start", total_items))

    def on_map_end(self, total_duration: float, ctx: CallbackContext) -> None:
        self.events.append(("map_end",))

    def on_map_item_start(self, item_index: int, ctx: CallbackContext) -> None:
        self.events.append(("map_item_start", item_index))

    def on_map_item_end(
        self, item_index: int, duration: float, ctx: CallbackContext
    ) -> None:
        self.events.append(("map_item_end", item_index))


def test_basic_callbacks():
    """Test that callbacks are triggered for basic execution."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    callback = TrackingCallback()
    engine = SeqEngine(callbacks=[callback])
    pipeline = Pipeline(nodes=[process], engine=engine)

    pipeline.run(inputs={"x": 5})

    # Check events
    event_types = [e[0] for e in callback.events]
    assert "pipeline_start" in event_types
    assert "node_start" in event_types
    assert "node_end" in event_types
    assert "pipeline_end" in event_types


def test_callbacks_with_multiple_nodes():
    """Test that callbacks are triggered for each node."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1

    callback = TrackingCallback()
    engine = SeqEngine(callbacks=[callback])
    pipeline = Pipeline(nodes=[double, add_one], engine=engine)

    pipeline.run(inputs={"x": 5})

    # Should have node_start/node_end for each node
    node_starts = [e for e in callback.events if e[0] == "node_start"]
    node_ends = [e for e in callback.events if e[0] == "node_end"]
    assert len(node_starts) == 2
    assert len(node_ends) == 2


def test_callbacks_with_map():
    """Test that map callbacks are triggered."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    callback = TrackingCallback()
    engine = SeqEngine(callbacks=[callback])
    pipeline = Pipeline(nodes=[process], engine=engine)

    pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")

    # Check map-specific events
    event_types = [e[0] for e in callback.events]
    assert "map_start" in event_types
    assert "map_item_start" in event_types
    assert "map_item_end" in event_types
    assert "map_end" in event_types

    # Should have 3 map items
    map_item_starts = [e for e in callback.events if e[0] == "map_item_start"]
    assert len(map_item_starts) == 3


def test_nested_pipeline_callback_inheritance():
    """Test that nested pipelines inherit parent's callbacks."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    # Inner pipeline with no callbacks specified
    inner = Pipeline(nodes=[double])

    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1

    callback = TrackingCallback()

    # Outer pipeline with callback - inner should inherit
    engine = SeqEngine(callbacks=[callback])
    outer = Pipeline(nodes=[inner.as_node(), add_one], engine=engine)

    outer.run(inputs={"x": 5})

    # Check for nested pipeline events
    event_types = [e[0] for e in callback.events]
    assert "nested_start" in event_types
    assert "nested_end" in event_types

    # Inner node events should be visible (through inheritance)
    # We should see node events for both inner and outer nodes
    node_starts = [e for e in callback.events if e[0] == "node_start"]
    assert len(node_starts) >= 2  # At least inner and outer nodes


def test_multiple_callbacks():
    """Test that multiple callbacks can be registered."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    callback1 = TrackingCallback()
    callback2 = TrackingCallback()

    engine = SeqEngine(callbacks=[callback1, callback2])
    pipeline = Pipeline(nodes=[process], engine=engine)

    pipeline.run(inputs={"x": 5})

    # Both callbacks should receive events
    assert len(callback1.events) > 0
    assert len(callback2.events) > 0

    # Events should be similar (same execution)
    event_types1 = [e[0] for e in callback1.events]
    event_types2 = [e[0] for e in callback2.events]
    assert event_types1 == event_types2


def test_callback_context_state_sharing():
    """Test that callbacks can share state via CallbackContext."""

    class SpanTracker(PipelineCallback):
        def on_node_start(
            self, node_id: str, inputs: dict, ctx: CallbackContext
        ) -> None:
            span_id = f"span_{node_id}"
            ctx.set("current_span", span_id)
            ctx.set(f"span:{node_id}", span_id)

    class MetricsTracker(PipelineCallback):
        def __init__(self):
            self.metrics = []

        def on_node_end(
            self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext
        ) -> None:
            span_id = ctx.get(f"span:{node_id}")
            self.metrics.append(
                {"node_id": node_id, "span_id": span_id, "duration": duration}
            )

    @node(output_name="result")
    def compute(x: int) -> int:
        return x + 5

    span_tracker = SpanTracker()
    metrics_tracker = MetricsTracker()

    engine = SeqEngine(callbacks=[span_tracker, metrics_tracker])
    pipeline = Pipeline(nodes=[compute], engine=engine)

    result = pipeline.run(inputs={"x": 10})

    assert result == {"result": 15}
    assert len(metrics_tracker.metrics) == 1
    assert metrics_tracker.metrics[0]["node_id"] == "compute"
    assert metrics_tracker.metrics[0]["span_id"] == "span_compute"
    assert metrics_tracker.metrics[0]["duration"] >= 0


def test_cache_hit_callback():
    """Test that on_node_cached is called for cache hits."""
    cache_events = []

    class CacheTracker(PipelineCallback):
        def on_node_cached(
            self, node_id: str, signature: str, ctx: CallbackContext
        ) -> None:
            cache_events.append(("cached", node_id, signature))

        def on_node_start(
            self, node_id: str, inputs: dict, ctx: CallbackContext
        ) -> None:
            cache_events.append(("executed", node_id))

    @node(output_name="result")
    def expensive_compute(x: int) -> int:
        return x**2

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = SeqEngine(cache=DiskCache(path=tmpdir), callbacks=[CacheTracker()])
        pipeline = Pipeline(nodes=[expensive_compute], engine=engine)

        # First run - should execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 25}
        assert ("executed", "expensive_compute") in cache_events
        assert not any(event[0] == "cached" for event in cache_events)

        # Second run - should be cached
        cache_events.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"result": 25}
        assert any(
            event[0] == "cached" and event[1] == "expensive_compute"
            for event in cache_events
        )
        assert not any(event[0] == "executed" for event in cache_events)


def test_error_handling_callback():
    """Test that on_error is called when nodes fail."""

    error_events = []

    class ErrorTracker(PipelineCallback):
        def on_error(
            self, node_id: str, error: Exception, ctx: CallbackContext
        ) -> None:
            error_events.append(
                {
                    "node_id": node_id,
                    "error_type": type(error).__name__,
                    "error_msg": str(error),
                }
            )

    @node(output_name="result")
    def failing_node(x: int) -> int:
        raise ValueError("Intentional failure")

    engine = SeqEngine(callbacks=[ErrorTracker()])
    pipeline = Pipeline(nodes=[failing_node], engine=engine)

    try:
        pipeline.run(inputs={"x": 10})
        assert False, "Should have raised error"
    except ValueError:
        pass

    assert len(error_events) == 1
    assert error_events[0]["node_id"] == "failing_node"
    assert error_events[0]["error_type"] == "ValueError"
    assert "Intentional failure" in error_events[0]["error_msg"]


def test_map_operation_with_cache():
    """Test that map operations properly track cache hits vs executions."""
    cache_events = []

    class MapCacheTracker(PipelineCallback):
        def on_node_cached(
            self, node_id: str, signature: str, ctx: CallbackContext
        ) -> None:
            # Track cached nodes during map
            if ctx.get("_in_map"):
                cache_events.append(("cached", node_id))

        def on_node_start(
            self, node_id: str, inputs: dict, ctx: CallbackContext
        ) -> None:
            # Track executed nodes during map
            if ctx.get("_in_map"):
                cache_events.append(("executed", node_id))

        def on_map_item_cached(
            self, item_index: int, signature: str, ctx: CallbackContext
        ) -> None:
            cache_events.append(("item_cached", item_index))

    @node(output_name="doubled")
    def double_cached(item: int) -> int:
        return item * 2

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = SeqEngine(cache=DiskCache(path=tmpdir), callbacks=[MapCacheTracker()])
        pipeline = Pipeline(nodes=[double_cached], engine=engine)

        # First run - all items execute
        result1 = pipeline.map(inputs={"item": [1, 2, 3]}, map_over="item")
        assert result1 == [
            {"doubled": 2},
            {"doubled": 4},
            {"doubled": 6},
        ]
        executed_count = sum(1 for e in cache_events if e[0] == "executed")
        assert executed_count == 3
        cached_count = sum(1 for e in cache_events if e[0] == "cached")
        assert cached_count == 0

        # Second run - all items cached
        cache_events.clear()
        result2 = pipeline.map(inputs={"item": [1, 2, 3]}, map_over="item")
        assert result2 == result1
        cached_count = sum(1 for e in cache_events if e[0] == "cached")
        assert cached_count == 3
        executed_count = sum(1 for e in cache_events if e[0] == "executed")
        assert executed_count == 0

        # Third run - partial overlap (items 1, 2 cached, item 4 executes)
        cache_events.clear()
        result3 = pipeline.map(inputs={"item": [1, 2, 4]}, map_over="item")
        assert result3 == [
            {"doubled": 2},
            {"doubled": 4},
            {"doubled": 8},
        ]
        cached_count = sum(1 for e in cache_events if e[0] == "cached")
        assert cached_count == 2
        executed_count = sum(1 for e in cache_events if e[0] == "executed")
        assert executed_count == 1


def test_incompatible_callback_error():
    """Test that an incompatible callback raises an error."""

    class DaftOnlyCallback(PipelineCallback):
        @property
        def supported_engines(self):
            return ["DaftEngine"]

    engine = SeqEngine(callbacks=[DaftOnlyCallback()])
    pipeline = Pipeline(nodes=[], engine=engine)

    try:
        pipeline.run(inputs={})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "is not compatible with engine SeqEngine" in str(e)

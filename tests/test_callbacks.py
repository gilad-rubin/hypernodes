"""Tests for callback behavior with SequentialEngine."""

from hypernodes import Pipeline, PipelineCallback, node
from hypernodes.callbacks import CallbackContext


class TrackingCallback(PipelineCallback):
    """Callback that tracks all events."""
    
    def __init__(self):
        self.events = []
    
    def on_pipeline_start(self, pipeline_id: str, ctx: CallbackContext) -> None:
        self.events.append(("pipeline_start", pipeline_id))
    
    def on_pipeline_end(self, pipeline_id: str, ctx: CallbackContext) -> None:
        self.events.append(("pipeline_end", pipeline_id))
    
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext) -> None:
        self.events.append(("node_start", node_id))
    
    def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext) -> None:
        self.events.append(("node_end", node_id))
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext) -> None:
        self.events.append(("node_cached", node_id))
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext) -> None:
        self.events.append(("nested_start", parent_id, child_pipeline_id))
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext) -> None:
        self.events.append(("nested_end", parent_id, child_pipeline_id))
    
    def on_map_start(self, pipeline_id: str, total_items: int, ctx: CallbackContext) -> None:
        self.events.append(("map_start", pipeline_id, total_items))
    
    def on_map_end(self, pipeline_id: str, total_items: int, ctx: CallbackContext) -> None:
        self.events.append(("map_end", pipeline_id, total_items))
    
    def on_map_item_start(self, pipeline_id: str, item_index: int, total_items: int, ctx: CallbackContext) -> None:
        self.events.append(("map_item_start", item_index))
    
    def on_map_item_end(self, pipeline_id: str, item_index: int, total_items: int, ctx: CallbackContext) -> None:
        self.events.append(("map_item_end", item_index))


def test_basic_callbacks():
    """Test that callbacks are triggered for basic execution."""
    
    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2
    
    callback = TrackingCallback()
    pipeline = Pipeline(nodes=[process], callbacks=[callback])
    
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
    pipeline = Pipeline(nodes=[double, add_one], callbacks=[callback])
    
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
    pipeline = Pipeline(nodes=[process], callbacks=[callback])
    
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
    outer = Pipeline(
        nodes=[inner.as_node(), add_one],
        callbacks=[callback]
    )
    
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
    
    pipeline = Pipeline(nodes=[process], callbacks=[callback1, callback2])
    
    pipeline.run(inputs={"x": 5})
    
    # Both callbacks should receive events
    assert len(callback1.events) > 0
    assert len(callback2.events) > 0
    
    # Events should be similar (same execution)
    event_types1 = [e[0] for e in callback1.events]
    event_types2 = [e[0] for e in callback2.events]
    assert event_types1 == event_types2


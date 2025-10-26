"""
Phase 4: Callbacks Tests

Test callback system for progress tracking, telemetry, and event handling.
"""

import tempfile

from hypernodes import DiskCache, Pipeline, node


def test_4_1_basic_progress_callback():
    """Test 4.1: Basic callback lifecycle events."""
    from hypernodes import PipelineCallback, CallbackContext
    
    events = []
    
    class TestProgressCallback(PipelineCallback):
        def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
            events.append(("start", node_id))
        
        def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
            events.append(("end", node_id, duration))
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    pipeline = Pipeline(
        nodes=[double, add_one],
        callbacks=[TestProgressCallback()]
    )
    
    result = pipeline.run(inputs={"x": 5})
    
    assert result["result"] == 11
    assert len(events) == 4
    assert events[0] == ("start", "double")
    assert events[1][0] == "end" and events[1][1] == "double"
    assert events[2] == ("start", "add_one")
    assert events[3][0] == "end" and events[3][1] == "add_one"
    assert all(events[i][2] >= 0 for i in [1, 3])


def test_4_2_pipeline_level_callbacks():
    """Test 4.2: Pipeline start/end events."""
    from hypernodes import PipelineCallback, CallbackContext
    
    events = []
    
    class PipelineTracker(PipelineCallback):
        def on_pipeline_start(self, pipeline_id: str, inputs: dict, ctx: CallbackContext):
            events.append(("pipeline_start", pipeline_id, list(inputs.keys())))
        
        def on_pipeline_end(self, pipeline_id: str, outputs: dict, duration: float, ctx: CallbackContext):
            events.append(("pipeline_end", pipeline_id, list(outputs.keys()), duration))
    
    @node(output_name="result")
    def compute(x: int) -> int:
        return x * 2
    
    pipeline = Pipeline(
        id="test_pipeline",
        nodes=[compute],
        callbacks=[PipelineTracker()]
    )
    
    result = pipeline.run(inputs={"x": 10})
    
    assert result["result"] == 20
    assert events[0] == ("pipeline_start", "test_pipeline", ["x"])
    assert events[1][0] == "pipeline_end"
    assert events[1][1] == "test_pipeline"
    assert events[1][2] == ["result"]
    assert events[1][3] >= 0


def test_4_3_multiple_callbacks():
    """Test 4.3: Multiple callbacks work together."""
    from hypernodes import PipelineCallback, CallbackContext
    
    progress_events = []
    log_events = []
    
    class ProgressTracker(PipelineCallback):
        def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
            progress_events.append(("progress_start", node_id))
        
        def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
            progress_events.append(("progress_end", node_id))
    
    class LoggingTracker(PipelineCallback):
        def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
            log_events.append(("log_start", node_id, list(inputs.keys())))
        
        def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
            log_events.append(("log_end", node_id, list(outputs.keys())))
    
    @node(output_name="result")
    def compute(x: int) -> int:
        return x * 3
    
    pipeline = Pipeline(
        nodes=[compute],
        callbacks=[ProgressTracker(), LoggingTracker()]
    )
    
    result = pipeline.run(inputs={"x": 7})
    
    assert result["result"] == 21
    assert len(progress_events) == 2
    assert len(log_events) == 2
    assert progress_events[0] == ("progress_start", "compute")
    assert log_events[0] == ("log_start", "compute", ["x"])
    assert progress_events[1] == ("progress_end", "compute")
    assert log_events[1] == ("log_end", "compute", ["result"])


def test_4_4_callback_context_state_sharing():
    """Test 4.4: Callbacks can share state via CallbackContext."""
    from hypernodes import PipelineCallback, CallbackContext
    
    class SpanTracker(PipelineCallback):
        def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
            span_id = f"span_{node_id}"
            ctx.set("current_span", span_id)
            ctx.set(f"span:{node_id}", span_id)
    
    class MetricsTracker(PipelineCallback):
        def __init__(self):
            self.metrics = []
        
        def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
            span_id = ctx.get(f"span:{node_id}")
            self.metrics.append({"node_id": node_id, "span_id": span_id, "duration": duration})
    
    @node(output_name="result")
    def compute(x: int) -> int:
        return x + 5
    
    span_tracker = SpanTracker()
    metrics_tracker = MetricsTracker()
    
    pipeline = Pipeline(nodes=[compute], callbacks=[span_tracker, metrics_tracker])
    
    result = pipeline.run(inputs={"x": 10})
    
    assert result["result"] == 15
    assert len(metrics_tracker.metrics) == 1
    assert metrics_tracker.metrics[0]["node_id"] == "compute"
    assert metrics_tracker.metrics[0]["span_id"] == "span_compute"
    assert metrics_tracker.metrics[0]["duration"] >= 0


def test_4_5_cache_hit_callback():
    """Test 4.5: on_node_cached is called for cache hits."""
    from hypernodes import PipelineCallback, CallbackContext
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_events = []
        
        class CacheTracker(PipelineCallback):
            def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
                cache_events.append(("cached", node_id, signature))
            
            def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
                cache_events.append(("executed", node_id))
        
        @node(output_name="result")
        def expensive_compute(x: int) -> int:
            return x ** 2
        
        pipeline = Pipeline(
            nodes=[expensive_compute],
            cache=DiskCache(path=tmpdir),
            callbacks=[CacheTracker()]
        )
        
        result1 = pipeline.run(inputs={"x": 5})
        assert result1["result"] == 25
        assert ("executed", "expensive_compute") in cache_events
        assert not any(event[0] == "cached" for event in cache_events)
        
        cache_events.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2["result"] == 25
        assert any(event[0] == "cached" and event[1] == "expensive_compute" for event in cache_events)
        assert not any(event[0] == "executed" for event in cache_events)


def test_4_6_error_handling_callback():
    """Test 4.6: on_error is called when nodes fail."""
    from hypernodes import PipelineCallback, CallbackContext
    
    error_events = []
    
    class ErrorTracker(PipelineCallback):
        def on_error(self, node_id: str, error: Exception, ctx: CallbackContext):
            error_events.append({
                "node_id": node_id,
                "error_type": type(error).__name__,
                "error_msg": str(error)
            })
    
    @node(output_name="result")
    def failing_node(x: int) -> int:
        raise ValueError("Intentional failure")
    
    pipeline = Pipeline(nodes=[failing_node], callbacks=[ErrorTracker()])
    
    try:
        pipeline.run(inputs={"x": 10})
        assert False, "Should have raised error"
    except ValueError:
        pass
    
    assert len(error_events) == 1
    assert error_events[0]["node_id"] == "failing_node"
    assert error_events[0]["error_type"] == "ValueError"
    assert "Intentional failure" in error_events[0]["error_msg"]


def test_4_7_map_operation_callbacks():
    """Test 4.7: Map-specific callback hooks."""
    from hypernodes import PipelineCallback, CallbackContext
    
    map_events = []
    
    class MapTracker(PipelineCallback):
        def on_map_start(self, total_items: int, ctx: CallbackContext):
            map_events.append(("map_start", total_items))
        
        def on_map_item_start(self, item_index: int, ctx: CallbackContext):
            map_events.append(("item_start", item_index))
        
        def on_map_item_end(self, item_index: int, duration: float, ctx: CallbackContext):
            map_events.append(("item_end", item_index))
        
        def on_map_end(self, total_duration: float, ctx: CallbackContext):
            map_events.append(("map_end",))
    
    @node(output_name="squared")
    def square(item: int) -> int:
        return item ** 2
    
    pipeline = Pipeline(nodes=[square], callbacks=[MapTracker()])
    
    result = pipeline.map(inputs={"item": [2, 3, 4]}, map_over="item")
    
    assert result["squared"] == [4, 9, 16]
    assert map_events[0] == ("map_start", 3)
    assert map_events[1] == ("item_start", 0)
    assert map_events[2] == ("item_end", 0)
    assert map_events[3] == ("item_start", 1)
    assert map_events[4] == ("item_end", 1)
    assert map_events[5] == ("item_start", 2)
    assert map_events[6] == ("item_end", 2)
    assert map_events[7] == ("map_end",)


def test_4_8_map_operation_with_cache():
    """Test 4.8: on_map_item_cached callback for cached map items."""
    from hypernodes import PipelineCallback, CallbackContext
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_events = []
        
        class MapCacheTracker(PipelineCallback):
            def on_map_item_cached(self, item_index: int, signature: str, ctx: CallbackContext):
                cache_events.append(("item_cached", item_index, signature))
            
            def on_map_item_start(self, item_index: int, ctx: CallbackContext):
                cache_events.append(("item_executed", item_index))
        
        @node(output_name="doubled")
        def double_cached(item: int) -> int:
            return item * 2
        
        pipeline = Pipeline(
            nodes=[double_cached],
            cache=DiskCache(path=tmpdir),
            callbacks=[MapCacheTracker()]
        )
        
        result1 = pipeline.map(inputs={"item": [1, 2, 3]}, map_over="item")
        assert result1["doubled"] == [2, 4, 6]
        executed_count = sum(1 for e in cache_events if e[0] == "item_executed")
        assert executed_count == 3
        
        cache_events.clear()
        result2 = pipeline.map(inputs={"item": [1, 2, 3]}, map_over="item")
        assert result2["doubled"] == [2, 4, 6]
        cached_count = sum(1 for e in cache_events if e[0] == "item_cached")
        assert cached_count == 3
        executed_count = sum(1 for e in cache_events if e[0] == "item_executed")
        assert executed_count == 0
        
        cache_events.clear()
        result3 = pipeline.map(inputs={"item": [1, 2, 4]}, map_over="item")
        assert result3["doubled"] == [2, 4, 8]
        cached_count = sum(1 for e in cache_events if e[0] == "item_cached")
        assert cached_count == 2
        executed_count = sum(1 for e in cache_events if e[0] == "item_executed")
        assert executed_count == 1

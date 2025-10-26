# Phase 4: Callbacks Tests

# Overview

Verify callback system for progress tracking, telemetry, and event handling. Tests progress from simple single callbacks to complex hierarchical scenarios with multiple callbacks and map operations.

---

## Test 4.1: Basic Progress Callback

**Goal:** Verify basic callback lifecycle events.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

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

events = []
result = [pipeline.run](http://pipeline.run)(x=5)

assert result["result"] == 11
assert len(events) == 4  # 2 nodes × 2 events (start/end)
assert events[0] == ("start", "double")
assert events[1][0] == "end" and events[1][1] == "double"
assert events[2] == ("start", "add_one")
assert events[3][0] == "end" and events[3][1] == "add_one"
assert all(events[i][2] >= 0 for i in [1, 3])  # Durations are non-negative
```

**Validates:**

- Callbacks receive lifecycle events
- Events in correct order
- Node IDs passed correctly
- Duration tracking

---

## Test 4.2: Pipeline-Level Callbacks

**Goal:** Verify pipeline start/end events.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

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

events = []
result = [pipeline.run](http://pipeline.run)(x=10)

assert result["result"] == 20
assert events[0] == ("pipeline_start", "test_pipeline", ["x"])
assert events[1][0] == "pipeline_end"
assert events[1][1] == "test_pipeline"
assert events[1][2] == ["result"]
assert events[1][3] >= 0  # Duration
```

**Validates:**

- Pipeline start/end hooks
- Input/output keys passed correctly
- Pipeline ID tracking
- Duration measurement

---

## Test 4.3: Multiple Callbacks

**Goal:** Verify multiple callbacks work together without interference.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node
import logging

logging.basicConfig(level=[logging.INFO](http://logging.INFO))

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

progress_events = []
log_events = []
result = [pipeline.run](http://pipeline.run)(x=7)

assert result["result"] == 21
assert len(progress_events) == 2
assert len(log_events) == 2
assert progress_events[0] == ("progress_start", "compute")
assert log_events[0] == ("log_start", "compute", ["x"])
assert progress_events[1] == ("progress_end", "compute")
assert log_events[1] == ("log_end", "compute", ["result"])
```

**Validates:**

- Multiple callbacks registered
- All callbacks receive events
- Callbacks don't interfere with each other
- Events maintain proper order

---

## Test 4.4: CallbackContext State Sharing

**Goal:** Verify callbacks can share state via CallbackContext.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

class SpanTracker(PipelineCallback):
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        span_id = f"span_{node_id}"
        ctx.set("current_span", span_id)
        ctx.set(f"span:{node_id}", span_id)

class MetricsTracker(PipelineCallback):
    def __init__(self):
        self.metrics = []
    
    def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        # Read span ID set by SpanTracker
        span_id = ctx.get(f"span:{node_id}")
        self.metrics.append({
            "node_id": node_id,
            "span_id": span_id,
            "duration": duration
        })

@node(output_name="result")
def compute(x: int) -> int:
    return x + 5

span_tracker = SpanTracker()
metrics_tracker = MetricsTracker()

pipeline = Pipeline(
    nodes=[compute],
    callbacks=[span_tracker, metrics_tracker]
)

result = [pipeline.run](http://pipeline.run)(x=10)

assert result["result"] == 15
assert len(metrics_tracker.metrics) == 1
assert metrics_tracker.metrics[0]["node_id"] == "compute"
assert metrics_tracker.metrics[0]["span_id"] == "span_compute"
assert metrics_tracker.metrics[0]["duration"] >= 0
```

**Validates:**

- CallbackContext state storage (set/get)
- State sharing between callbacks
- Context preserves data across callback invocations

---

## Test 4.5: Cache Hit Callback

**Goal:** Verify on_node_cached is called for cache hits.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

cache_events = []

class CacheTracker(PipelineCallback):
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        cache_events.append(("cached", node_id, signature))
    
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        cache_events.append(("executed", node_id))

@node(output_name="result", cache=True)
def expensive_compute(x: int) -> int:
    return x ** 2

pipeline = Pipeline(
    nodes=[expensive_compute],
    callbacks=[CacheTracker()]
)

# First run - should execute
cache_events = []
result1 = [pipeline.run](http://pipeline.run)(x=5)
assert result1["result"] == 25
assert ("executed", "expensive_compute") in cache_events
assert not any(event[0] == "cached" for event in cache_events)

# Second run - should use cache
cache_events = []
result2 = [pipeline.run](http://pipeline.run)(x=5)
assert result2["result"] == 25
assert any(event[0] == "cached" and event[1] == "expensive_compute" for event in cache_events)
assert not any(event[0] == "executed" for event in cache_events)
```

**Validates:**

- on_node_cached called for cache hits
- Cache signature passed correctly
- Distinction between execution and cache retrieval

---

## Test 4.6: Error Handling Callback

**Goal:** Verify on_error is called when nodes fail.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

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

pipeline = Pipeline(
    nodes=[failing_node],
    callbacks=[ErrorTracker()]
)

error_events = []
try:
    [pipeline.run](http://pipeline.run)(x=10)
    assert False, "Should have raised error"
except ValueError:
    pass

assert len(error_events) == 1
assert error_events[0]["node_id"] == "failing_node"
assert error_events[0]["error_type"] == "ValueError"
assert "Intentional failure" in error_events[0]["error_msg"]
```

**Validates:**

- on_error called when node fails
- Error information passed correctly
- Callback called before exception propagates

---

## Test 4.7: Nested Pipeline Callbacks - Basic

**Goal:** Verify callbacks work with simple nested pipelines.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

events = []

class HierarchyTracker(PipelineCallback):
    def on_pipeline_start(self, pipeline_id: str, inputs: dict, ctx: CallbackContext):
        events.append(("pipeline_start", pipeline_id, ctx.depth))
    
    def on_pipeline_end(self, pipeline_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        events.append(("pipeline_end", pipeline_id, ctx.depth))
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        events.append(("nested_start", parent_id, child_pipeline_id, ctx.depth))
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        events.append(("nested_end", parent_id, child_pipeline_id, ctx.depth))

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

inner_pipeline = Pipeline(
    id="inner",
    nodes=[double]
)

outer_pipeline = Pipeline(
    id="outer",
    nodes=[inner_pipeline],
    callbacks=[HierarchyTracker()]
)

events = []
result = outer_[pipeline.run](http://pipeline.run)(x=5)

assert result["doubled"] == 10
assert events[0] == ("pipeline_start", "outer", 0)
assert events[1] == ("nested_start", "outer", "inner", 0)
assert events[2] == ("pipeline_start", "inner", 1)
assert events[3] == ("pipeline_end", "inner", 1)
assert events[4][0] == "nested_end"
assert events[5] == ("pipeline_end", "outer", 0)
```

**Validates:**

- Nested pipeline hooks called
- Depth tracking in context
- Parent/child relationship preserved
- Proper hook ordering

---

## Test 4.8: Nested Pipeline Callbacks - Hierarchy Path

**Goal:** Verify hierarchy_path tracking through nested pipelines.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

paths = []

class PathTracker(PipelineCallback):
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        paths.append(("node", node_id, list(ctx.hierarchy_path)))

@node(output_name="x2")
def times_two(x: int) -> int:
    return x * 2

@node(output_name="x3")
def times_three(x2: int) -> int:
    return x2 * 3

inner = Pipeline(
    id="inner",
    nodes=[times_three]
)

middle = Pipeline(
    id="middle",
    nodes=[times_two, inner]
)

outer = Pipeline(
    id="outer",
    nodes=[middle],
    callbacks=[PathTracker()]
)

paths = []
result = [outer.run](http://outer.run)(x=2)

assert result["x3"] == 12
assert paths[0] == ("node", "times_two", ["outer", "middle"])
assert paths[1] == ("node", "times_three", ["outer", "middle", "inner"])
```

**Validates:**

- hierarchy_path builds correctly
- Path includes all ancestor pipeline IDs
- Path updates correctly when entering/exiting nested pipelines

---

## Test 4.9: Map Operation Callbacks - Basic

**Goal:** Verify map-specific callback hooks.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

map_events = []

class MapTracker(PipelineCallback):
    def on_map_start(self, node_id: str, total_items: int, ctx: CallbackContext):
        map_events.append(("map_start", node_id, total_items))
    
    def on_map_item_start(self, node_id: str, item_index: int, item: any, ctx: CallbackContext):
        map_events.append(("item_start", node_id, item_index))
    
    def on_map_item_end(self, node_id: str, item_index: int, duration: float, ctx: CallbackContext):
        map_events.append(("item_end", node_id, item_index))
    
    def on_map_end(self, node_id: str, total_duration: float, ctx: CallbackContext):
        map_events.append(("map_end", node_id))

@node(output_name="squared", map_mode="items")
def square(item: int) -> int:
    return item ** 2

pipeline = Pipeline(
    nodes=[square],
    callbacks=[MapTracker()]
)

map_events = []
result = [pipeline.run](http://pipeline.run)(item=[2, 3, 4])

assert result["squared"] == [4, 9, 16]
assert map_events[0] == ("map_start", "square", 3)
assert map_events[1] == ("item_start", "square", 0)
assert map_events[2] == ("item_end", "square", 0)
assert map_events[3] == ("item_start", "square", 1)
assert map_events[4] == ("item_end", "square", 1)
assert map_events[5] == ("item_start", "square", 2)
assert map_events[6] == ("item_end", "square", 2)
assert map_events[7] == ("map_end", "square")
```

**Validates:**

- on_map_start called with total item count
- on_map_item_start/end called for each item
- Item indices passed correctly
- on_map_end called after all items

---

## Test 4.10: Map Operation with Cache

**Goal:** Verify on_map_item_cached callback for cached map items.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

cache_events = []

class MapCacheTracker(PipelineCallback):
    def on_map_item_cached(self, node_id: str, item_index: int, signature: str, ctx: CallbackContext):
        cache_events.append(("item_cached", node_id, item_index, signature))
    
    def on_map_item_start(self, node_id: str, item_index: int, item: any, ctx: CallbackContext):
        cache_events.append(("item_executed", node_id, item_index))

@node(output_name="doubled", map_mode="items", cache=True)
def double_cached(item: int) -> int:
    return item * 2

pipeline = Pipeline(
    nodes=[double_cached],
    callbacks=[MapCacheTracker()]
)

# First run - all items executed
cache_events = []
result1 = [pipeline.run](http://pipeline.run)(item=[1, 2, 3])
assert result1["doubled"] == [2, 4, 6]
executed_count = sum(1 for e in cache_events if e[0] == "item_executed")
assert executed_count == 3

# Second run - all items cached
cache_events = []
result2 = [pipeline.run](http://pipeline.run)(item=[1, 2, 3])
assert result2["doubled"] == [2, 4, 6]
cached_count = sum(1 for e in cache_events if e[0] == "item_cached")
assert cached_count == 3
executed_count = sum(1 for e in cache_events if e[0] == "item_executed")
assert executed_count == 0

# Partial cache - items 1 and 2 cached, item 4 new
cache_events = []
result3 = [pipeline.run](http://pipeline.run)(item=[1, 2, 4])
assert result3["doubled"] == [2, 4, 8]
cached_count = sum(1 for e in cache_events if e[0] == "item_cached")
assert cached_count == 2  # Items 1 and 2
executed_count = sum(1 for e in cache_events if e[0] == "item_executed")
assert executed_count == 1  # Item 4
```

**Validates:**

- on_map_item_cached called for cache hits
- Cache/execution distinction per item
- Partial cache scenarios handled correctly

---

## Test 4.11: Callback Inheritance - Basic

**Goal:** Verify callbacks inherit to nested pipelines.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

events = []

class InheritanceTracker(PipelineCallback):
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        events.append(("node_start", node_id, ctx.current_pipeline_id))

@node(output_name="x2")
def double(x: int) -> int:
    return x * 2

@node(output_name="x3")
def triple(x2: int) -> int:
    return x2 * 3

# Child pipeline with NO callbacks specified - should inherit
child = Pipeline(
    id="child",
    nodes=[triple]
)

# Parent pipeline with callback
parent = Pipeline(
    id="parent",
    nodes=[double, child],
    callbacks=[InheritanceTracker()]
)

events = []
result = [parent.run](http://parent.run)(x=5)

assert result["x3"] == 30
# Both parent and child nodes should have events
assert ("node_start", "double", "parent") in events
assert ("node_start", "triple", "child") in events
```

**Validates:**

- Callbacks inherit to child pipelines
- Inherited callbacks work in nested context
- Context correctly identifies current pipeline

---

## Test 4.12: Callback Inheritance - Override

**Goal:** Verify child pipelines can override parent callbacks.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

parent_events = []
child_events = []

class ParentCallback(PipelineCallback):
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        parent_events.append(("parent", node_id))

class ChildCallback(PipelineCallback):
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        child_events.append(("child", node_id))

@node(output_name="a")
def node_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def node_b(a: int) -> int:
    return a + 2

# Child with explicit callbacks - should NOT inherit
child = Pipeline(
    id="child",
    nodes=[node_b],
    callbacks=[ChildCallback()]
)

parent = Pipeline(
    id="parent",
    nodes=[node_a, child],
    callbacks=[ParentCallback()]
)

parent_events = []
child_events = []
result = [parent.run](http://parent.run)(x=10)

assert result["b"] == 13
# Parent callback should only see parent nodes
assert ("parent", "node_a") in parent_events
assert ("parent", "node_b") not in parent_events
# Child callback should only see child nodes
assert ("child", "node_b") in child_events
assert ("child", "node_a") not in child_events
```

**Validates:**

- Explicit callbacks override inheritance
- Parent callbacks don't propagate when overridden
- Each pipeline uses its own callbacks

---

## Test 4.13: Callback Inheritance - Empty Override

**Goal:** Verify empty callback list disables all callbacks.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

events = []

class TrackerCallback(PipelineCallback):
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        events.append(node_id)

@node(output_name="a")
def node_a(x: int) -> int:
    return x * 2

@node(output_name="b")
def node_b(a: int) -> int:
    return a + 5

# Quiet child with empty callbacks - disables all callbacks
quiet_child = Pipeline(
    id="quiet",
    nodes=[node_b],
    callbacks=[]  # Explicit empty list
)

parent = Pipeline(
    id="parent",
    nodes=[node_a, quiet_child],
    callbacks=[TrackerCallback()]
)

events = []
result = [parent.run](http://parent.run)(x=3)

assert result["b"] == 11
# Only parent node tracked
assert "node_a" in events
assert "node_b" not in events
```

**Validates:**

- Empty callback list disables inheritance
- Selective callback disabling for sub-pipelines

---

## Test 4.14: Complex Nested Hierarchy with Multiple Callbacks

**Goal:** Test complex nested structure with multiple callbacks sharing state.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

span_log = []
metrics_log = []

class SpanCallback(PipelineCallback):
    def on_pipeline_start(self, pipeline_id: str, inputs: dict, ctx: CallbackContext):
        parent_span = ctx.get("current_span")
        span = f"span_{pipeline_id}"
        ctx.set("current_span", span)
        ctx.set(f"pipeline_span:{pipeline_id}", span)
        span_log.append(("start", pipeline_id, parent_span, ctx.depth))
    
    def on_pipeline_end(self, pipeline_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        span = ctx.get(f"pipeline_span:{pipeline_id}")
        span_log.append(("end", pipeline_id, span, ctx.depth))
        # Restore parent span
        if ctx.parent_pipeline_id:
            parent_span = ctx.get(f"pipeline_span:{ctx.parent_pipeline_id}")
            ctx.set("current_span", parent_span)

class MetricsCallback(PipelineCallback):
    def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        span = ctx.get("current_span")
        metrics_log.append({
            "node": node_id,
            "span": span,
            "path": " > ".join(ctx.hierarchy_path),
            "depth": ctx.depth
        })

@node(output_name="a")
def node_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def node_b(a: int) -> int:
    return a * 2

@node(output_name="c")
def node_c(b: int) -> int:
    return b + 10

level2 = Pipeline(id="level2", nodes=[node_c])
level1 = Pipeline(id="level1", nodes=[node_b, level2])
root = Pipeline(
    id="root",
    nodes=[node_a, level1],
    callbacks=[SpanCallback(), MetricsCallback()]
)

span_log = []
metrics_log = []
result = [root.run](http://root.run)(x=5)

assert result["c"] == 22  # ((5+1)*2)+10

# Verify span hierarchy
assert span_log[0] == ("start", "root", None, 0)
assert span_log[1] == ("start", "level1", "span_root", 1)
assert span_log[2] == ("start", "level2", "span_level1", 2)

# Verify metrics attached to correct spans
assert metrics_log[0]["node"] == "node_a"
assert metrics_log[0]["span"] == "span_root"
assert metrics_log[0]["path"] == "root"
assert metrics_log[0]["depth"] == 0

assert metrics_log[1]["node"] == "node_b"
assert metrics_log[1]["span"] == "span_level1"
assert metrics_log[1]["path"] == "root > level1"
assert metrics_log[1]["depth"] == 1

assert metrics_log[2]["node"] == "node_c"
assert metrics_log[2]["span"] == "span_level2"
assert metrics_log[2]["path"] == "root > level1 > level2"
assert metrics_log[2]["depth"] == 2
```

**Validates:**

- Multiple callbacks share context state correctly
- Hierarchical span relationships maintained
- Metrics correctly associated with spans
- Context state properly restored on exit
- Deep nesting (3 levels) works correctly

---

## Test 4.15: Pipeline Metadata Access

**Goal:** Verify callbacks can access pipeline metadata via context.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

metadata_log = []

class MetadataCallback(PipelineCallback):
    def on_pipeline_start(self, pipeline_id: str, inputs: dict, ctx: CallbackContext):
        metadata = ctx.get_pipeline_metadata(pipeline_id)
        metadata_log.append({
            "pipeline": pipeline_id,
            "total_nodes": metadata.get("total_nodes"),
            "node_ids": metadata.get("node_ids")
        })

@node(output_name="a")
def node_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def node_b(a: int) -> int:
    return a * 2

@node(output_name="c")
def node_c(b: int) -> int:
    return b + 5

pipeline = Pipeline(
    id="test",
    nodes=[node_a, node_b, node_c],
    callbacks=[MetadataCallback()]
)

metadata_log = []
result = [pipeline.run](http://pipeline.run)(x=10)

assert result["c"] == 27
assert len(metadata_log) == 1
assert metadata_log[0]["pipeline"] == "test"
assert metadata_log[0]["total_nodes"] == 3
assert metadata_log[0]["node_ids"] == ["node_a", "node_b", "node_c"]
```

**Validates:**

- get_pipeline_metadata returns correct info
- Total node count available
- Node IDs list available
- Useful for progress bar initialization

---

## Test 4.16: All Callback Hooks Integration

**Goal:** Comprehensive test covering all callback methods in a realistic scenario.

```python
from pipeline_system import PipelineCallback, CallbackContext, Pipeline, node

class ComprehensiveCallback(PipelineCallback):
    def __init__(self):
        [self.events](http://self.events) = []
    
    def on_pipeline_start(self, pipeline_id: str, inputs: dict, ctx: CallbackContext):
        [self.events](http://self.events).append(("pipeline_start", pipeline_id))
    
    def on_pipeline_end(self, pipeline_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        [self.events](http://self.events).append(("pipeline_end", pipeline_id))
    
    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        [self.events](http://self.events).append(("node_start", node_id))
    
    def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        [self.events](http://self.events).append(("node_end", node_id))
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        [self.events](http://self.events).append(("node_cached", node_id))
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        [self.events](http://self.events).append(("nested_start", parent_id, child_pipeline_id))
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        [self.events](http://self.events).append(("nested_end", parent_id, child_pipeline_id))
    
    def on_map_start(self, node_id: str, total_items: int, ctx: CallbackContext):
        [self.events](http://self.events).append(("map_start", node_id, total_items))
    
    def on_map_item_start(self, node_id: str, item_index: int, item: any, ctx: CallbackContext):
        [self.events](http://self.events).append(("map_item_start", node_id, item_index))
    
    def on_map_item_end(self, node_id: str, item_index: int, duration: float, ctx: CallbackContext):
        [self.events](http://self.events).append(("map_item_end", node_id, item_index))
    
    def on_map_end(self, node_id: str, total_duration: float, ctx: CallbackContext):
        [self.events](http://self.events).append(("map_end", node_id))
    
    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext):
        [self.events](http://self.events).append(("error", node_id, type(error).__name__))

@node(output_name="processed", map_mode="items", cache=True)
def process_item(item: int) -> int:
    return item * 2

@node(output_name="total")
def sum_results(processed: list) -> int:
    return sum(processed)

inner = Pipeline(
    id="inner",
    nodes=[process_item, sum_results]
)

@node(output_name="outer_result")
def finalize(total: int) -> int:
    return total + 100

outer = Pipeline(
    id="outer",
    nodes=[inner, finalize],
    callbacks=[ComprehensiveCallback()]
)

callback = outer.callbacks[0]
result1 = [outer.run](http://outer.run)(item=[1, 2, 3])

assert result1["outer_result"] == 112  # (2+4+6) + 100

# Verify all major event types captured
event_types = [e[0] for e in [callback.events](http://callback.events)]
assert "pipeline_start" in event_types
assert "pipeline_end" in event_types
assert "node_start" in event_types
assert "node_end" in event_types
assert "nested_start" in event_types
assert "nested_end" in event_types
assert "map_start" in event_types
assert "map_item_start" in event_types
assert "map_item_end" in event_types
assert "map_end" in event_types

# Run again to test caching
[callback.events](http://callback.events) = []
result2 = [outer.run](http://outer.run)(item=[1, 2, 3])
event_types = [e[0] for e in [callback.events](http://callback.events)]
assert "node_cached" in event_types  # Map items should be cached
```

**Validates:**

- All callback methods work together
- Realistic pipeline with nesting + maps + caching
- Event ordering is logical
- No hook is missed in complex scenarios
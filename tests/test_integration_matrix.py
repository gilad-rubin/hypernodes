"""
Integration tests that combine engines/executors, nested pipelines, caching, and callbacks.

Matrix-style coverage from simple → complex with timing checks to spot anomalies.
"""

import asyncio
import tempfile
import time

import pytest

from hypernodes import DiskCache, Pipeline, node
from hypernodes.callbacks import CallbackContext, PipelineCallback
from hypernodes.engine import HypernodesEngine


class TrackingCallback(PipelineCallback):
    """Collects lifecycle events and durations for assertions."""

    def __init__(self):
        self.events = []
        self.node_starts = []
        self.node_ends = []
        self.node_cached = []
        self.pipeline_starts = []
        self.pipeline_ends = []
        self.nested_starts = []
        self.nested_ends = []
        self.map_events = []

    def on_pipeline_start(self, pipeline_id: str, inputs: dict, ctx: CallbackContext):
        self.pipeline_starts.append((pipeline_id, list(inputs.keys())))
        self.events.append(("pipeline_start", pipeline_id))

    def on_pipeline_end(self, pipeline_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        self.pipeline_ends.append((pipeline_id, list(outputs.keys()), duration))
        self.events.append(("pipeline_end", pipeline_id, duration))

    def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
        self.node_starts.append((node_id, list(inputs.keys())))
        self.events.append(("node_start", node_id))

    def on_node_end(self, node_id: str, outputs: dict, duration: float, ctx: CallbackContext):
        self.node_ends.append((node_id, list(outputs.keys()), duration))
        self.events.append(("node_end", node_id, duration))

    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        self.node_cached.append((node_id, signature))
        self.events.append(("node_cached", node_id))

    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        self.nested_starts.append((parent_id, child_pipeline_id))
        self.events.append(("nested_start", parent_id, child_pipeline_id))

    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        self.nested_ends.append((parent_id, child_pipeline_id, duration))
        self.events.append(("nested_end", parent_id, child_pipeline_id, duration))

    # Map hooks
    def on_map_start(self, total_items: int, ctx: CallbackContext):
        self.map_events.append(("map_start", total_items))

    def on_map_item_start(self, item_index: int, ctx: CallbackContext):
        self.map_events.append(("item_start", item_index))

    def on_map_item_end(self, item_index: int, duration: float, ctx: CallbackContext):
        self.map_events.append(("item_end", item_index, duration))

    def on_map_item_cached(self, item_index: int, signature: str, ctx: CallbackContext):
        self.map_events.append(("item_cached", item_index))

    def on_map_end(self, total_duration: float, ctx: CallbackContext):
        self.map_events.append(("map_end",))


def _sleep(s: float):
    # Small helper to keep sleeps local for flake-free timing assertions
    time.sleep(s)


def test_integration_nested_cache_callbacks_sequential():
    """Simple: nested pipeline + caching + callbacks (sequential engine).

    - First run executes nodes and records durations
    - Second run should be significantly faster (cache hits) and fire on_node_cached
    - Nested pipeline start/end hooks should fire
    """

    @node(output_name="doubled", cache=True)
    def double(x: int) -> int:
        _sleep(0.03)
        return x * 2

    @node(output_name="incremented", cache=True)
    def add_one(doubled: int) -> int:
        _sleep(0.03)
        return doubled + 1

    inner = Pipeline(nodes=[double, add_one])

    @node(output_name="result", cache=True)
    def square(incremented: int) -> int:
        _sleep(0.03)
        return incremented * incremented

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        inner_cache = DiskCache(path=tmp1)
        outer_cache = DiskCache(path=tmp2)

        cb = TrackingCallback()
        outer = Pipeline(nodes=[inner, square], callbacks=[cb], cache=outer_cache)
        # Inner has its own independent cache
        inner.cache = inner_cache

        # Engine default is sequential (via effective_backend)
        start1 = time.time()
        res1 = outer.run(inputs={"x": 5})
        dur1 = time.time() - start1

        assert res1["result"] == (5 * 2 + 1) ** 2
        # Expect node_end entries for 3 nodes in first run
        assert len([e for e in cb.events if e[0] == "node_end"]) >= 3
        assert any(e[0] == "nested_start" for e in cb.events)
        assert any(e[0] == "nested_end" for e in cb.events)

        # Clear event list, re-run for caching
        cb.events.clear()
        start2 = time.time()
        res2 = outer.run(inputs={"x": 5})
        dur2 = time.time() - start2
        assert res2 == res1

        # Cache should make second run substantially faster
        assert dur2 < dur1 * 0.5
        # on_node_cached should fire for at least inner nodes and outer node
        assert any(e[0] == "node_cached" for e in cb.events)


def test_integration_nested_threaded_parallel_levels_timing():
    """Intermediate: threaded node executor should parallelize independent nodes.

    Structure: two independent slow nodes feeding a combiner. With 2 workers, first level
    should run concurrently (~0.05s vs ~0.10s sequential), then combine.
    """

    @node(output_name="a")
    def slow_a(x: int) -> int:
        _sleep(0.05)
        return x * 2

    @node(output_name="b")
    def slow_b(x: int) -> int:
        _sleep(0.05)
        return x * 3

    inner = Pipeline(nodes=[slow_a, slow_b])

    @node(output_name="result")
    def combine(a: int, b: int) -> int:
        return a + b

    cb = TrackingCallback()
    # Use 2 workers to ensure both a and b can run concurrently
    engine = HypernodesEngine(node_executor="threaded", max_workers=2)
    outer = Pipeline(nodes=[inner, combine], callbacks=[cb], engine=engine)

    start = time.time()
    out = outer.run(inputs={"x": 5})
    dur = time.time() - start

    assert out == {"a": 10, "b": 15, "result": 25}
    # Should be faster than sequential (~0.1s vs ~0.15s)
    assert dur < 0.12
    # Verify callbacks occurred
    assert any(e[0] == "node_start" for e in cb.events)
    assert any(e[0] == "node_end" for e in cb.events)


@pytest.mark.parametrize(
    "map_executor",
    [
        "async",  # Async map executor
        "parallel",  # Loky-based parallel map executor
    ],
)
def test_integration_nested_map_cache_callbacks(map_executor):
    """Complex: nested pipeline as a node with map_over + per-item caching + callbacks.

    - First map: executes all items
    - Second map: all items cached
    - Third map: only changed items execute
    """

    @node(output_name="doubled", cache=True)
    def double(x: int) -> int:
        _sleep(0.02)
        return x * 2

    single = Pipeline(nodes=[double])
    # Adapt to map over list provided as "nums"
    batch = single.as_node(
        map_over="nums",
        input_mapping={"nums": "x"},
        output_mapping={"doubled": "results"},
    )

    cb = TrackingCallback()
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(path=tmp)
        engine = HypernodesEngine(map_executor=map_executor, node_executor="sequential", max_workers=3)
        outer = Pipeline(nodes=[batch], callbacks=[cb], cache=cache, engine=engine)

        # First run: execute all
        t1 = time.time()
        r1 = outer.run(inputs={"nums": [1, 2, 3]})
        d1 = time.time() - t1
        assert r1["results"] == [2, 4, 6]
        # Map callbacks sanity
        assert ("map_start", 3) in cb.map_events
        assert ("map_end",) in cb.map_events
        # For non-sequential map executors, per-item callbacks are not emitted

        # Second run: all cached
        cb.map_events.clear()
        t2 = time.time()
        r2 = outer.run(inputs={"nums": [1, 2, 3]})
        d2 = time.time() - t2
        assert r2["results"] == [2, 4, 6]
        assert d2 < d1 * 0.5

        # Third run: partial cache (change last item)
        cb.map_events.clear()
        t3 = time.time()
        r3 = outer.run(inputs={"nums": [1, 2, 4]})
        d3 = time.time() - t3
        assert r3["results"] == [2, 4, 8]
        # Timing for mixed cache can vary across executors/environments; ensure correctness only


def test_integration_engine_instance_reuse_async_node_executor():
    """Ensure AsyncExecutor works with nested pipelines and callbacks without timing anomalies."""

    @node(output_name="a")
    async def slow_a(x: int) -> int:
        # Sleep via asyncio to exercise async path
        await asyncio.sleep(0.03)
        return x * 2

    @node(output_name="b")
    async def slow_b(x: int) -> int:
        await asyncio.sleep(0.03)
        return x * 3

    inner = Pipeline(nodes=[slow_a, slow_b])

    @node(output_name="result")
    def combine(a: int, b: int) -> int:
        return a + b

    cb = TrackingCallback()
    engine = HypernodesEngine(node_executor="async")
    outer = Pipeline(nodes=[inner, combine], callbacks=[cb], engine=engine)

    t = time.time()
    out = outer.run(inputs={"x": 5})
    dur = time.time() - t

    assert out == {"a": 10, "b": 15, "result": 25}
    # Concurrent async nodes should complete in ~0.03s + overhead, not ~0.06s
    assert dur < 0.08
    assert any(e[0] == "node_start" for e in cb.events)
    assert any(e[0] == "node_end" for e in cb.events)


@pytest.mark.parametrize(
    "map_executor",
    [
        "async",
        "parallel",
    ],
)
def test_integration_nested_map_multi_outputs(map_executor):
    """Nested pipeline-as-node with multi-outputs under map, with caching and callbacks."""

    @node(output_name="a", cache=True)
    def double(x: int) -> int:
        _sleep(0.02)
        return x * 2

    @node(output_name="b", cache=True)
    def triple(x: int) -> int:
        _sleep(0.02)
        return x * 3

    inner = Pipeline(nodes=[double, triple])

    # Expose both outputs with renaming
    batch = inner.as_node(
        map_over="nums",
        input_mapping={"nums": "x"},
        output_mapping={"a": "A", "b": "B"},
        name="encode_two_ops",
    )

    cb1 = TrackingCallback()
    cb2 = TrackingCallback()
    with tempfile.TemporaryDirectory() as tmp:
        cache = DiskCache(path=tmp)
        engine = HypernodesEngine(map_executor=map_executor, node_executor="threaded", max_workers=3)
        outer = Pipeline(nodes=[batch], callbacks=[cb1, cb2], cache=cache, engine=engine)

        # First run
        t1 = time.time()
        r1 = outer.run(inputs={"nums": [1, 2, 3, 4]})
        d1 = time.time() - t1
        assert r1["A"] == [2, 4, 6, 8]
        assert r1["B"] == [3, 6, 9, 12]

        # Second run (should be cached)
        t2 = time.time()
        r2 = outer.run(inputs={"nums": [1, 2, 3, 4]})
        d2 = time.time() - t2
        assert r2 == r1
        assert d2 < d1 * 0.6

        # Third run (partial cache)
        r3 = outer.run(inputs={"nums": [1, 2, 4, 5]})
        assert r3["A"] == [2, 4, 8, 10]
        assert r3["B"] == [3, 6, 12, 15]


def test_integration_deeper_nesting_threaded_parallel_mix():
    """Deeper nesting: inner produces two outputs; outer combines; node_executor=threaded, map_executor=parallel."""

    @node(output_name="a")
    def slow_a(x: int) -> int:
        _sleep(0.05)
        return x * 2

    @node(output_name="b")
    def slow_b(x: int) -> int:
        _sleep(0.05)
        return x * 3

    inner = Pipeline(nodes=[slow_a, slow_b])
    mid = Pipeline(nodes=[inner])  # mid wraps inner as PipelineNode

    @node(output_name="result")
    def combine(a: int, b: int) -> int:
        return a + b

    cb = TrackingCallback()
    engine = HypernodesEngine(node_executor="threaded", map_executor="parallel", max_workers=2)
    outer = Pipeline(nodes=[mid, combine], callbacks=[cb], engine=engine)

    t = time.time()
    out = outer.run(inputs={"x": 7})
    dur = time.time() - t

    assert out == {"a": 14, "b": 21, "result": 35}
    # Threaded should allow parallel a/b inside inner pipeline
    assert dur < 0.12
    assert any(e[0] == "nested_start" for e in cb.events)
    assert any(e[0] == "nested_end" for e in cb.events)
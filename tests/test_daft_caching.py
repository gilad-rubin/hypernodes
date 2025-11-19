"""Tests for caching behavior with DaftEngine."""

import pytest
import time
from typing import List, Dict, Any

# Skip if daft is not installed
daft = pytest.importorskip("daft")

from hypernodes import Pipeline, node, DiskCache
from hypernodes.integrations.daft.engine import DaftEngine

@pytest.fixture
def disk_cache(tmp_path):
    """Create a temporary disk cache."""
    return DiskCache(str(tmp_path / "cache"))

def test_daft_basic_caching(disk_cache):
    """Test that caching prevents re-execution of nodes in DaftEngine."""
    
    # Use a mutable list to track execution count
    execution_count = {"count": 0}
    
    @node(output_name="y")
    def slow_function(x: int) -> int:
        execution_count["count"] += 1
        return x * 2
        
    pipeline = Pipeline(nodes=[slow_function])
    
    # Enable caching on the engine
    engine = DaftEngine(cache=disk_cache)
    
    # First run - should execute
    result1 = engine.run(pipeline, {"x": 10})
    assert result1["y"] == 20
    assert execution_count["count"] == 1
    
    # Second run - should hit cache
    result2 = engine.run(pipeline, {"x": 10})
    assert result2["y"] == 20
    assert execution_count["count"] == 1  # Count should not increase

def test_daft_map_caching(disk_cache):
    """Test that caching works with map operations in DaftEngine."""
    
    execution_count = {"count": 0}
    
    @node(output_name="y")
    def process(x: int) -> int:
        execution_count["count"] += 1
        return x + 1
        
    pipeline = Pipeline(nodes=[process])
    engine = DaftEngine(cache=disk_cache)
    
    inputs = {"x": [1, 2, 3]}
    
    # First run
    results1 = engine.map(pipeline, inputs, map_over="x")
    assert len(results1) == 3
    assert results1[0]["y"] == 2
    # In Daft map, it might execute once as a batch or N times depending on implementation
    # But we just want to ensure second run doesn't increase it
    initial_count = execution_count["count"]
    assert initial_count > 0
    
    # Second run - should hit cache
    results2 = engine.map(pipeline, inputs, map_over="x")
    assert len(results2) == 3
    assert results2[0]["y"] == 2
    assert execution_count["count"] == initial_count

def test_daft_cache_invalidation(disk_cache):
    """Test that cache is invalidated when inputs change."""
    
    execution_count = {"count": 0}
    
    @node(output_name="y")
    def process(x: int) -> int:
        execution_count["count"] += 1
        return x * 2
        
    pipeline = Pipeline(nodes=[process])
    engine = DaftEngine(cache=disk_cache)
    
    # Run with x=10
    engine.run(pipeline, {"x": 10})
    assert execution_count["count"] == 1
    
    # Run with x=20 - should re-execute
    engine.run(pipeline, {"x": 20})
    assert execution_count["count"] == 2
    
    # Run with x=10 again - should hit cache
    engine.run(pipeline, {"x": 10})
    assert execution_count["count"] == 2

def test_daft_partial_caching(disk_cache):
    """Test partial pipeline caching (A cached, B re-executed)."""
    
    counts = {"a": 0, "b": 0}
    
    @node(output_name="a_out")
    def node_a(x: int) -> int:
        counts["a"] += 1
        return x + 1
        
    @node(output_name="b_out")
    def node_b(a_out: int) -> int:
        counts["b"] += 1
        return a_out * 2
        
    pipeline = Pipeline(nodes=[node_a, node_b])
    engine = DaftEngine(cache=disk_cache)
    
    # Run 1
    engine.run(pipeline, {"x": 1})
    assert counts["a"] == 1
    assert counts["b"] == 1
    
    # Run 2 - full cache hit
    engine.run(pipeline, {"x": 1})
    assert counts["a"] == 1
    assert counts["b"] == 1
    
    # Run 3 - invalidate B only? 
    # In DaftEngine, we can't easily invalidate just B without changing code/inputs.
    # But if we change input to B (which comes from A), A must change.
    # So let's test diamond pattern or similar where we can vary things.
    # Actually, let's just verify that if we disable cache for B, it re-runs.
    
    node_b.cache = False
    # Note: node.cache modification might persist, so be careful or reset
    
    # Run 4 - A should hit cache, B should run
    engine.run(pipeline, {"x": 1})
    assert counts["a"] == 1
    assert counts["b"] == 2

def test_daft_map_incremental_caching(disk_cache):
    """Test that map mode caches items individually (per-item caching)."""
    execution_count = {"count": 0}
    
    @node(output_name="y")
    def process(x: int) -> int:
        execution_count["count"] += 1
        return x + 1
    
    pipeline = Pipeline(nodes=[process])
    engine = DaftEngine(cache=disk_cache)
    
    # First run: [1, 2, 3]
    results1 = engine.map(pipeline, {"x": [1, 2, 3]}, map_over="x")
    assert len(results1) == 3
    assert results1[0]["y"] == 2
    assert results1[1]["y"] == 3
    assert results1[2]["y"] == 4
    first_count = execution_count["count"]
    assert first_count == 3  # All three items executed
    
    # Second run: [1, 2, 4] - should only execute item with x=4
    results2 = engine.map(pipeline, {"x": [1, 2, 4]}, map_over="x")
    assert len(results2) == 3
    assert results2[0]["y"] == 2  # Cached
    assert results2[1]["y"] == 3  # Cached
    assert results2[2]["y"] == 5  # New computation
    # Count should increase by 1 (only item with x=4)
    assert execution_count["count"] == first_count + 1
    
    # Third run: [1, 2, 3] - all should be cached
    results3 = engine.map(pipeline, {"x": [1, 2, 3]}, map_over="x")
    assert len(results3) == 3
    assert execution_count["count"] == first_count + 1  # No increase

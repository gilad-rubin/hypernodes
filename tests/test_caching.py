"""Tests for caching behavior with SequentialEngine."""

import tempfile
import time

from hypernodes import DiskCache, Pipeline, node


def test_basic_caching():
    """Test that caching prevents re-execution of nodes."""
    
    execution_count = []
    
    @node(output_name="result", cache=True)
    def slow_function(x: int) -> int:
        execution_count.append(1)
        return x * 2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        pipeline = Pipeline(nodes=[slow_function], cache=cache)
        
        # First run - should execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 10}
        assert len(execution_count) == 1
        
        # Second run - should use cache
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"result": 10}
        assert len(execution_count) == 1  # Still 1, not executed again


def test_cache_invalidation_on_input_change():
    """Test that cache is invalidated when inputs change."""
    
    execution_count = []
    
    @node(output_name="result", cache=True)
    def process(x: int) -> int:
        execution_count.append(1)
        return x * 2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        pipeline = Pipeline(nodes=[process], cache=cache)
        
        # First input
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 10}
        assert len(execution_count) == 1
        
        # Different input - should execute again
        result2 = pipeline.run(inputs={"x": 10})
        assert result2 == {"result": 20}
        assert len(execution_count) == 2
        
        # Original input again - should use cache
        result3 = pipeline.run(inputs={"x": 5})
        assert result3 == {"result": 10}
        assert len(execution_count) == 2  # Still 2


def test_selective_caching():
    """Test that cache=False prevents caching for specific nodes."""
    
    execution_count = {"cached": [], "uncached": []}
    
    @node(output_name="cached_result", cache=True)
    def cached_function(x: int) -> int:
        execution_count["cached"].append(1)
        return x * 2
    
    @node(output_name="uncached_result", cache=False)
    def uncached_function(cached_result: int) -> int:
        execution_count["uncached"].append(1)
        return cached_result + 1
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        pipeline = Pipeline(nodes=[cached_function, uncached_function], cache=cache)
        
        # First run
        pipeline.run(inputs={"x": 5})
        assert len(execution_count["cached"]) == 1
        assert len(execution_count["uncached"]) == 1
        
        # Second run - cached node uses cache, uncached runs again
        pipeline.run(inputs={"x": 5})
        assert len(execution_count["cached"]) == 1  # Used cache
        assert len(execution_count["uncached"]) == 2  # Ran again


def test_nested_pipeline_cache_inheritance():
    """Test that nested pipelines inherit parent's cache."""
    
    execution_count = []
    
    @node(output_name="doubled", cache=True)
    def double(x: int) -> int:
        execution_count.append(1)
        return x * 2
    
    # Inner pipeline with no cache specified
    inner = Pipeline(nodes=[double])
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        
        # Outer pipeline with cache - inner should inherit
        outer = Pipeline(
            nodes=[inner.as_node(), add_one],
            cache=cache
        )
        
        # First run - inner node executes
        result1 = outer.run(inputs={"x": 5})
        assert result1 == {"doubled": 10, "result": 11}
        assert len(execution_count) == 1
        
        # Second run - inner node should use cache (inherited)
        result2 = outer.run(inputs={"x": 5})
        assert result2 == {"doubled": 10, "result": 11}
        assert len(execution_count) == 1  # Still 1, used cache


def test_cache_with_map():
    """Test that caching works correctly with map operations."""
    
    execution_count = []
    
    @node(output_name="result", cache=True)
    def expensive_operation(x: int) -> int:
        execution_count.append(x)
        time.sleep(0.01)  # Simulate expensive operation
        return x * 2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        pipeline = Pipeline(nodes=[expensive_operation], cache=cache)
        
        # First map - all items execute
        results1 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results1 == [
            {"result": 2},
            {"result": 4},
            {"result": 6},
        ]
        assert len(execution_count) == 3
        
        # Second map with same items - all use cache
        results2 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results2 == results1
        assert len(execution_count) == 3  # Still 3, used cache
        
        # Third map with partial overlap - only new item executes
        results3 = pipeline.map(inputs={"x": [2, 3, 4]}, map_over="x")
        assert results3 == [
            {"result": 4},
            {"result": 6},
            {"result": 8},
        ]
        assert len(execution_count) == 4  # Only item 4 executed
        assert execution_count[-1] == 4


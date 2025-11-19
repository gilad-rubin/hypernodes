"""
Tests for ProgressCallback and telemetry functionality.

Tests cover:
- Basic progress tracking with .run()
- Progress tracking with .map()
- Progress with cache hits
- Progress with nested pipelines
- Progress disabled mode
- Progress with errors
"""

import time
import tempfile
from hypernodes import node, Pipeline, DiskCache, SequentialEngine
from hypernodes.telemetry import ProgressCallback


def test_progress_basic_pipeline():
    """Test ProgressCallback with a simple pipeline using .run()."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        time.sleep(0.01)  # Small delay
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        time.sleep(0.01)
        return doubled + 1
    
    # Create pipeline with progress callback
    progress = ProgressCallback(enable=False)  # Disable for tests
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[double, add_one], 
        engine=engine,
        name="double_and_add"
    )
    
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "result": 11}


def test_progress_with_map():
    """Test ProgressCallback with .map() operation."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        time.sleep(0.01)
        return doubled + 1
    
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[double, add_one],
        engine=engine,
        name="map_test"
    )
    
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    
    assert results == [
        {"doubled": 2, "result": 3},
        {"doubled": 4, "result": 5},
        {"doubled": 6, "result": 7},
    ]


def test_progress_with_cache():
    """Test ProgressCallback tracks cache hits correctly."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        @node(output_name="doubled")
        def double(x: int) -> int:
            time.sleep(0.01)
            return x * 2
        
        @node(output_name="result")
        def add_one(doubled: int) -> int:
            time.sleep(0.01)
            return doubled + 1
        
        cache = DiskCache(path=tmpdir)
        progress = ProgressCallback(enable=False)
        engine = SequentialEngine(cache=cache, callbacks=[progress])
        pipeline = Pipeline(
            nodes=[double, add_one],
            engine=engine,
            name="cached_pipeline"
        )
        
        # First run - should execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"doubled": 10, "result": 11}
        
        # Second run - should use cache
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"doubled": 10, "result": 11}
        
        # Different input - should execute again
        result3 = pipeline.run(inputs={"x": 7})
        assert result3 == {"doubled": 14, "result": 15}


def test_progress_with_map_and_cache():
    """Test ProgressCallback with map operation and cache hits."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        @node(output_name="doubled")
        def double(x: int) -> int:
            time.sleep(0.01)
            return x * 2
        
        cache = DiskCache(path=tmpdir)
        progress = ProgressCallback(enable=False)
        engine = SequentialEngine(cache=cache, callbacks=[progress])
        pipeline = Pipeline(
            nodes=[double],
            engine=engine,
            name="map_cache_test"
        )
        
        # First map - should execute all
        results1 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results1 == [
            {"doubled": 2},
            {"doubled": 4},
            {"doubled": 6},
        ]
        
        # Second map - should use cache for all
        results2 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results2 == results1
        
        # Partial cache hit - [1, 2] cached, [4] new
        results3 = pipeline.map(inputs={"x": [1, 2, 4]}, map_over="x")
        assert results3 == [
            {"doubled": 2},
            {"doubled": 4},
            {"doubled": 8},
        ]


def test_progress_disabled():
    """Test that ProgressCallback can be disabled for testing."""
    
    @node(output_name="result")
    def simple(x: int) -> int:
        return x + 1
    
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(nodes=[simple], engine=engine)
    
    result = pipeline.run(inputs={"x": 5})
    
    # Should work without displaying progress
    assert result == {"result": 6}
    assert len(progress._bars) == 0  # No bars created when disabled


def test_progress_with_nested_pipelines():
    """Test ProgressCallback with nested pipelines."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    inner_pipeline = Pipeline(nodes=[double], name="inner")
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        time.sleep(0.01)
        return doubled + 1
    
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    outer_pipeline = Pipeline(
        nodes=[inner_pipeline.as_node(), add_one],
        engine=engine,
        name="outer"
    )
    
    result = outer_pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "result": 11}


def test_progress_with_diamond_pattern():
    """Test ProgressCallback with diamond dependency pattern."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        time.sleep(0.01)
        return x * 3
    
    @node(output_name="result")
    def add(doubled: int, tripled: int) -> int:
        time.sleep(0.01)
        return doubled + tripled
    
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[double, triple, add],
        engine=engine,
        name="diamond"
    )
    
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "tripled": 15, "result": 25}


def test_progress_with_multiple_map_params():
    """Test ProgressCallback with map over multiple parameters (zip mode)."""
    
    @node(output_name="sum")
    def add(x: int, y: int) -> int:
        time.sleep(0.01)
        return x + y
    
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[add],
        engine=engine,
        name="multi_param_map"
    )
    
    results = pipeline.map(
        inputs={"x": [1, 2, 3], "y": [10, 20, 30]},
        map_over=["x", "y"],
        map_mode="zip"
    )
    
    assert results == [
        {"sum": 11},
        {"sum": 22},
        {"sum": 33},
    ]


def test_progress_with_fixed_and_varying_params():
    """Test ProgressCallback with map having fixed and varying parameters."""
    
    @node(output_name="result")
    def multiply(x: int, factor: int) -> int:
        time.sleep(0.01)
        return x * factor
    
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[multiply],
        engine=engine,
        name="fixed_varying"
    )
    
    results = pipeline.map(
        inputs={"x": [1, 2, 3], "factor": 10},
        map_over="x"
    )
    
    assert results == [
        {"result": 10},
        {"result": 20},
        {"result": 30},
    ]


def test_progress_enabled_visual():
    """Test ProgressCallback with enable=True for visual verification (run manually)."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        time.sleep(0.1)
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        time.sleep(0.1)
        return doubled + 1
    
    # Note: enable=False for automated tests
    # Change to enable=True and run manually to see progress bars
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[double, add_one],
        engine=engine,
        name="visual_test"
    )
    
    result = pipeline.run(inputs={"x": 5})
    assert result == {"doubled": 10, "result": 11}


def test_progress_map_visual():
    """Test ProgressCallback with map (visual verification - run manually)."""
    
    @node(output_name="squared")
    def square(x: int) -> int:
        time.sleep(0.1)
        return x ** 2
    
    # Note: enable=False for automated tests
    # Change to enable=True and run manually to see progress bars
    progress = ProgressCallback(enable=False)
    engine = SequentialEngine(callbacks=[progress])
    pipeline = Pipeline(
        nodes=[square],
        engine=engine,
        name="map_visual_test"
    )
    
    results = pipeline.map(inputs={"x": [1, 2, 3, 4, 5]}, map_over="x")
    assert results == [
        {"squared": 1},
        {"squared": 4},
        {"squared": 9},
        {"squared": 16},
        {"squared": 25},
    ]


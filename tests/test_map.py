"""Tests for map operations with SequentialEngine (default)."""

from hypernodes import Pipeline, node


def test_simple_map():
    """Test basic map operation over a list."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    pipeline = Pipeline(nodes=[double])
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    
    assert results == [
        {"doubled": 2},
        {"doubled": 4},
        {"doubled": 6},
    ]


def test_map_with_fixed_params():
    """Test map with both varying and fixed parameters."""
    
    @node(output_name="result")
    def multiply(x: int, factor: int) -> int:
        return x * factor
    
    pipeline = Pipeline(nodes=[multiply])
    results = pipeline.map(
        inputs={"x": [1, 2, 3], "factor": 10},
        map_over="x"
    )
    
    assert results == [
        {"result": 10},
        {"result": 20},
        {"result": 30},
    ]


def test_map_zip_mode():
    """Test map with zip mode (parallel iteration)."""
    
    @node(output_name="result")
    def add(x: int, y: int) -> int:
        return x + y
    
    pipeline = Pipeline(nodes=[add])
    results = pipeline.map(
        inputs={"x": [1, 2, 3], "y": [10, 20, 30]},
        map_over=["x", "y"],
        map_mode="zip"
    )
    
    assert results == [
        {"result": 11},
        {"result": 22},
        {"result": 33},
    ]


def test_map_product_mode():
    """Test map with product mode (all combinations)."""
    
    @node(output_name="result")
    def multiply(x: int, y: int) -> int:
        return x * y
    
    pipeline = Pipeline(nodes=[multiply])
    results = pipeline.map(
        inputs={"x": [2, 3], "y": [10, 100]},
        map_over=["x", "y"],
        map_mode="product"
    )
    
    # 2*10, 2*100, 3*10, 3*100
    assert results == [
        {"result": 20},
        {"result": 200},
        {"result": 30},
        {"result": 300},
    ]


def test_map_sequential_nodes():
    """Test map over pipeline with multiple nodes."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    pipeline = Pipeline(nodes=[double, add_one])
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    
    assert results == [
        {"doubled": 2, "result": 3},
        {"doubled": 4, "result": 5},
        {"doubled": 6, "result": 7},
    ]


def test_map_with_selective_output():
    """Test map with selective output filtering."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="tripled")
    def triple(doubled: int) -> int:
        return doubled * 3
    
    pipeline = Pipeline(nodes=[double, triple])
    results = pipeline.map(
        inputs={"x": [1, 2, 3]},
        map_over="x",
        output_name="tripled"
    )
    
    assert results == [
        {"tripled": 6},
        {"tripled": 12},
        {"tripled": 18},
    ]


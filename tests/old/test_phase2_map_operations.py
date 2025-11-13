"""
Phase 2: Map Operations Tests

Test pipeline.map() for batch processing with various parameter configurations
and execution patterns.
"""
from hypernodes import node, Pipeline


def test_2_1_map_over_single_parameter():
    """Test 2.1: Map over single parameter.
    
    Validates:
    - pipeline.map() executes over collection
    - inputs is a dictionary with list values for varying parameters
    - map_over accepts single string for single parameter
    - Results returned as lists
    - Order preserved
    """
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one])
    
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert results == {"result": [2, 3, 4]}


def test_2_2_map_over_two_sequential_nodes():
    """Test 2.2: Two sequential nodes with map.
    
    Validates:
    - Map works with multi-node pipelines
    - All intermediate results returned as lists
    - Dependencies resolved per item
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    pipeline = Pipeline(nodes=[double, add_one])
    
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert results == {"doubled": [2, 4, 6], "result": [3, 5, 7]}


def test_2_3_map_with_diamond_pattern():
    """Test 2.3: Map with diamond dependency pattern.
    
    Validates:
    - Complex DAG patterns work with map
    - Each item executed independently
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3
    
    @node(output_name="result")
    def add(doubled: int, tripled: int) -> int:
        return doubled + tripled
    
    pipeline = Pipeline(nodes=[double, triple, add])
    
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert results == {"doubled": [2, 4, 6], "tripled": [3, 6, 9], "result": [5, 10, 15]}


def test_2_4_map_with_fixed_and_varying_parameters():
    """Test 2.4: Map with fixed and varying parameters.
    
    Validates:
    - Fixed parameters used for all items
    - Only parameters in map_over vary
    - Correct behavior with mixed varying/fixed inputs
    """
    @node(output_name="result")
    def multiply(x: int, factor: int) -> int:
        return x * factor
    
    pipeline = Pipeline(nodes=[multiply])
    
    results = pipeline.map(inputs={"x": [1, 2, 3], "factor": 10}, map_over="x")
    assert results == {"result": [10, 20, 30]}


def test_2_5_empty_collection():
    """Test 2.5: Map handles empty input gracefully.
    
    Validates:
    - Empty collection returns empty results
    - No errors raised
    """
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one])
    
    results = pipeline.map(inputs={"x": []}, map_over="x")
    assert results == {"result": []}


def test_2_6_map_with_multiple_parameters_zip_mode():
    """Test 2.6: Map with multiple parameters using zip mode (default).
    
    Validates:
    - Multiple map_over parameters work
    - Zip mode processes corresponding items together
    - Default map_mode is "zip"
    - Order preserved
    """
    @node(output_name="result")
    def process(id: int, text: str) -> str:
        return f"{id}: {text.upper()}"
    
    pipeline = Pipeline(nodes=[process])
    
    results = pipeline.map(
        inputs={"id": [1, 2, 3], "text": ["hello", "world", "test"]},
        map_over=["id", "text"]
    )
    assert results == {"result": ["1: HELLO", "2: WORLD", "3: TEST"]}


def test_2_7_zip_mode_mismatched_lengths_error():
    """Test 2.7: Zip mode raises error when list lengths don't match.
    
    Validates:
    - Zip mode validates list lengths
    - Clear error message when lengths don't match
    - Prevents silent bugs from length mismatches
    """
    @node(output_name="result")
    def process(id: int, text: str) -> str:
        return f"{id}: {text.upper()}"
    
    pipeline = Pipeline(nodes=[process])
    
    # Should raise error: lists have different lengths
    try:
        results = pipeline.map(
            inputs={"id": [1, 2, 3], "text": ["hello", "world"]},  # Mismatched: 3 vs 2
            map_over=["id", "text"]
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "length" in str(e).lower()


def test_2_8_map_with_product_mode():
    """Test 2.8: Map with multiple parameters using product mode.
    
    Validates:
    - Product mode creates all combinations
    - Lists can be of different lengths
    - Correct number of executions (6 = 2 × 3)
    - Results in correct order
    """
    @node(output_name="result")
    def multiply(x: int, y: int) -> int:
        return x * y
    
    pipeline = Pipeline(nodes=[multiply])
    
    results = pipeline.map(
        inputs={"x": [1, 2], "y": [10, 20, 30]},
        map_over=["x", "y"],
        map_mode="product"
    )
    # (1,10), (1,20), (1,30), (2,10), (2,20), (2,30)
    assert results == {"result": [10, 20, 30, 20, 40, 60]}


def test_2_9_zip_mode_with_fixed_parameter():
    """Test 2.9: Zip mode with some parameters fixed and others varying.
    
    Validates:
    - Zip mode with subset of parameters varying
    - Fixed parameters used in all executions
    - Correct behavior with mixed varying/fixed inputs
    """
    @node(output_name="result")
    def format_message(id: int, text: str, prefix: str) -> str:
        return f"{prefix}{id}: {text}"
    
    pipeline = Pipeline(nodes=[format_message])
    
    results = pipeline.map(
        inputs={"id": [1, 2], "text": ["hello", "world"], "prefix": "MSG-"},
        map_over=["id", "text"],
        map_mode="zip"  # Explicit, but this is default
    )
    assert results == {"result": ["MSG-1: hello", "MSG-2: world"]}

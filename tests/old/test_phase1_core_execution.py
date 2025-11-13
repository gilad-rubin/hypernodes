"""
Phase 1: Core Execution Tests

Test basic function decoration and single-node pipeline execution through
progressively complex dependency patterns.
"""
from hypernodes import node, Pipeline


def test_1_1_single_node_with_simple_inputs():
    """Test 1.1: Single node with simple inputs.
    
    Validates:
    - @node decorator with output_name parameter works
    - Pipeline construction with single function
    - Default backend (LocalBackend with sequential execution) is used
    - pipeline.run() executes and returns correct output
    - Output is returned as a dictionary with correct key
    - Result contains ONLY outputs, not inputs
    """
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one])
    result = pipeline.run(inputs={"x": 5})
    
    # Result should contain ONLY outputs, not inputs
    assert result == {"result": 6}
    assert "x" not in result


def test_1_2_two_sequential_nodes():
    """Test 1.2: Two sequential nodes.
    
    Validates:
    - Dependency resolution based on parameter names
    - Sequential execution order (double → add_one)
    - Multiple outputs returned in result dictionary
    - Intermediate results accessible
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    pipeline = Pipeline(nodes=[double, add_one])
    result = pipeline.run(inputs={"x": 5})
    
    # Only outputs, not inputs
    assert result == {"doubled": 10, "result": 11}
    assert "x" not in result


def test_1_3_three_nodes_with_linear_dependencies():
    """Test 1.3: Three nodes with linear dependencies.
    
    Validates:
    - Longer dependency chains work correctly
    - Each intermediate result is computed and available
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    pipeline = Pipeline(nodes=[double, add_one, square])
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "incremented": 11, "result": 121}
    assert "x" not in result


def test_1_4_diamond_dependency_pattern():
    """Test 1.4: Diamond dependency pattern.
    
    Validates:
    - Multiple nodes can depend on same input
    - Node with multiple dependencies waits for all
    - Correct execution with diamond pattern
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
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "tripled": 15, "result": 25}
    assert "x" not in result


def test_1_5_multiple_independent_inputs():
    """Test 1.5: Multiple independent inputs.
    
    Validates:
    - Multiple input parameters work
    - Multiple nodes can use same inputs
    - All outputs computed correctly
    """
    @node(output_name="sum")
    def add(x: int, y: int) -> int:
        return x + y
    
    @node(output_name="product")
    def multiply(x: int, y: int) -> int:
        return x * y
    
    @node(output_name="result")
    def combine(sum: int, product: int) -> int:
        return sum + product
    
    pipeline = Pipeline(nodes=[add, multiply, combine])
    result = pipeline.run(inputs={"x": 5, "y": 3})
    
    assert result == {"sum": 8, "product": 15, "result": 23}
    assert "x" not in result
    assert "y" not in result


def test_1_6_simple_nested_pipeline():
    """Test 1.6: Simple nested pipeline.
    
    Validates:
    - Pipeline used as node
    - Outputs from nested pipeline available to outer pipeline
    - Dependencies resolved across pipeline boundaries
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner_pipeline = Pipeline(nodes=[double, add_one])
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    # Wrap inner pipeline as a node
    outer_pipeline = Pipeline(nodes=[inner_pipeline.as_node(), square])
    result = outer_pipeline.run(inputs={"x": 5})
    
    # All outputs from inner and outer pipeline
    assert result == {"doubled": 10, "incremented": 11, "result": 121}
    assert "x" not in result

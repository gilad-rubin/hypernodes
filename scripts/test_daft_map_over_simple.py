#!/usr/bin/env python3
"""
Simple test for DaftBackend with .as_node(map_over=...)

Tests the new map_over support using simple data types.
"""

from typing import List
from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend


# ==================== Simple Nodes ====================
@node(output_name="numbers")
def create_numbers(count: int) -> List[int]:
    """Create a list of numbers."""
    return list(range(count))


@node(output_name="doubled")
def double_number(number: int) -> int:
    """Double a single number."""
    return number * 2


@node(output_name="sum")
def sum_numbers(all_doubled: List[int]) -> int:
    """Sum all doubled numbers."""
    return sum(all_doubled)


# ==================== Test with DaftBackend ====================
def test_daft_backend_map_over():
    """Test DaftBackend with .as_node(map_over=...)."""
    print("\n" + "="*70)
    print("TEST: DaftBackend with .as_node(map_over=...)")
    print("="*70)
    
    # Single-item pipeline
    double_single = Pipeline(
        nodes=[double_number],
        name="double_single"
    )
    
    # Create mapped node
    double_mapped = double_single.as_node(
        input_mapping={"numbers": "number"},
        output_mapping={"doubled": "all_doubled"},
        map_over="numbers",
        name="double_mapped"
    )
    
    # Full pipeline with DaftBackend
    pipeline = Pipeline(
        nodes=[
            create_numbers,
            double_mapped,
            sum_numbers,
        ],
        backend=DaftBackend(show_plan=True),
        name="test_daft_map"
    )
    
    inputs = {"count": 5}
    
    print(f"\nInputs: {inputs}")
    print("Running pipeline...")
    
    result = pipeline.run(inputs=inputs, output_name="sum")
    
    print(f"\nResult: {result}")
    
    # Verify: sum of [0*2, 1*2, 2*2, 3*2, 4*2] = 0 + 2 + 4 + 6 + 8 = 20
    expected = sum(i * 2 for i in range(5))
    assert result["sum"] == expected, f"Expected {expected}, got {result['sum']}"
    
    print(f"✅ Test passed! Got expected result: {expected}")


def test_nested_map_over():
    """Test nested map operations."""
    print("\n" + "="*70)
    print("TEST: Nested map_over operations")
    print("="*70)
    
    @node(output_name="numbers")
    def create_numbers(count: int) -> List[int]:
        return list(range(count))
    
    @node(output_name="squared")
    def square(number: int) -> int:
        return number ** 2
    
    @node(output_name="result")
    def sum_all(all_squared: List[int]) -> int:
        return sum(all_squared)
    
    # Pipeline: square each number
    square_single = Pipeline(nodes=[square], name="square_single")
    square_mapped = square_single.as_node(
        input_mapping={"numbers": "number"},
        output_mapping={"squared": "all_squared"},
        map_over="numbers",
        name="square_mapped"
    )
    
    pipeline = Pipeline(
        nodes=[create_numbers, square_mapped, sum_all],
        backend=DaftBackend(),
        name="test_nested"
    )
    
    result = pipeline.run(inputs={"count": 4}, output_name="result")
    
    # 0^2 + 1^2 + 2^2 + 3^2 = 0 + 1 + 4 + 9 = 14
    expected = sum(i**2 for i in range(4))
    assert result["result"] == expected
    
    print(f"✅ Test passed! Sum of squares: {result['result']}")


def test_with_multiple_inputs():
    """Test map_over with shared inputs."""
    print("\n" + "="*70)
    print("TEST: map_over with shared inputs")
    print("="*70)
    
    @node(output_name="numbers")
    def create_numbers(count: int) -> List[int]:
        return list(range(count))
    
    @node(output_name="scaled")
    def scale(number: int, factor: int) -> int:
        return number * factor
    
    @node(output_name="total")
    def sum_all(all_scaled: List[int]) -> int:
        return sum(all_scaled)
    
    # Pipeline with shared input 'factor'
    scale_single = Pipeline(nodes=[scale], name="scale_single")
    scale_mapped = scale_single.as_node(
        input_mapping={"numbers": "number"},
        output_mapping={"scaled": "all_scaled"},
        map_over="numbers",
        name="scale_mapped"
    )
    
    pipeline = Pipeline(
        nodes=[create_numbers, scale_mapped, sum_all],
        backend=DaftBackend(),
        name="test_shared_inputs"
    )
    
    result = pipeline.run(inputs={"count": 3, "factor": 10}, output_name="total")
    
    # (0 * 10) + (1 * 10) + (2 * 10) = 0 + 10 + 20 = 30
    expected = sum(i * 10 for i in range(3))
    assert result["total"] == expected
    
    print(f"✅ Test passed! Scaled sum: {result['total']}")


# ==================== Main ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DaftBackend .as_node(map_over=...) Support - Test Suite")
    print("="*70)
    
    test_daft_backend_map_over()
    test_nested_map_over()
    test_with_multiple_inputs()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print("\nDaftBackend now supports .as_node(map_over=...)!")
    print("Strategy: explode() → transform → groupby().agg(list())")

#!/usr/bin/env python3
"""
Test script for Daft code generation feature.

This script demonstrates the code generation mode of DaftEngine by:
1. Creating a simple test pipeline
2. Generating Daft code from it
3. Saving the code to a file
4. Optionally comparing with actual execution
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Simple test nodes
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2


@node(output_name="tripled")
def triple(x: int) -> int:
    """Triple the input."""
    return x * 3


@node(output_name="result")
def add(doubled: int, tripled: int) -> int:
    """Add doubled and tripled values."""
    return doubled + tripled


# More complex example with stateful object
class Multiplier:
    """Stateful multiplier class."""
    
    def __init__(self, factor: int):
        self.factor = factor
    
    def multiply(self, value: int) -> int:
        return value * self.factor


@node(output_name="multiplied")
def multiply_with_object(x: int, multiplier: Multiplier) -> int:
    """Multiply using stateful object."""
    return multiplier.multiply(x)


@node(output_name="final_result")
def add_results(result: int, multiplied: int) -> int:
    """Add both results."""
    return result + multiplied


def test_simple_pipeline():
    """Test code generation for a simple pipeline."""
    print("\n" + "=" * 70)
    print("TEST 1: Simple Pipeline Code Generation")
    print("=" * 70)
    
    # Create pipeline
    pipeline = Pipeline(
        nodes=[double, triple, add],
        name="simple_pipeline"
    )
    
    # Generate Daft code
    inputs = {"x": 5}
    generated_code = pipeline.show_daft_code(inputs=inputs)
    
    print("\nGenerated Daft Code:")
    print("-" * 70)
    print(generated_code)
    print("-" * 70)
    
    # Save to file
    output_file = "scripts/generated_simple_pipeline.py"
    with open(output_file, "w") as f:
        f.write(generated_code)
    print(f"\n✅ Code saved to: {output_file}")
    
    return generated_code


def test_stateful_pipeline():
    """Test code generation for a pipeline with stateful objects."""
    print("\n" + "=" * 70)
    print("TEST 2: Stateful Pipeline Code Generation")
    print("=" * 70)
    
    # Create pipeline with stateful object
    pipeline = Pipeline(
        nodes=[double, triple, add, multiply_with_object, add_results],
        name="stateful_pipeline"
    )
    
    # Generate Daft code
    inputs = {
        "x": 5,
        "multiplier": Multiplier(factor=10)
    }
    generated_code = pipeline.show_daft_code(inputs=inputs)
    
    print("\nGenerated Daft Code:")
    print("-" * 70)
    print(generated_code)
    print("-" * 70)
    
    # Save to file
    output_file = "scripts/generated_stateful_pipeline.py"
    with open(output_file, "w") as f:
        f.write(generated_code)
    print(f"\n✅ Code saved to: {output_file}")
    
    return generated_code


def test_map_pipeline():
    """Test code generation for a pipeline with map operations."""
    print("\n" + "=" * 70)
    print("TEST 3: Map Pipeline Code Generation")
    print("=" * 70)
    
    # Create single-item pipeline
    single_pipeline = Pipeline(
        nodes=[double],
        name="single_double"
    )
    
    # Wrap as mapped node
    double_many = single_pipeline.as_node(
        input_mapping={"numbers": "x"},
        output_mapping={"doubled": "doubled_numbers"},
        map_over="numbers",
        name="double_many"
    )
    
    # Create final pipeline
    @node(output_name="total")
    def sum_all(doubled_numbers: list) -> int:
        """Sum all doubled numbers."""
        return sum(doubled_numbers)
    
    pipeline = Pipeline(
        nodes=[double_many, sum_all],
        name="map_pipeline"
    )
    
    # Generate Daft code
    inputs = {"numbers": [1, 2, 3, 4, 5]}
    generated_code = pipeline.show_daft_code(inputs=inputs)
    
    print("\nGenerated Daft Code:")
    print("-" * 70)
    print(generated_code)
    print("-" * 70)
    
    # Save to file
    output_file = "scripts/generated_map_pipeline.py"
    with open(output_file, "w") as f:
        f.write(generated_code)
    print(f"\n✅ Code saved to: {output_file}")
    
    return generated_code


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DAFT CODE GENERATION TESTS")
    print("=" * 70)
    
    try:
        # Run tests
        test_simple_pipeline()
        test_stateful_pipeline()
        test_map_pipeline()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - scripts/generated_simple_pipeline.py")
        print("  - scripts/generated_stateful_pipeline.py")
        print("  - scripts/generated_map_pipeline.py")
        print("\nYou can now:")
        print("  1. Review the generated code to understand Daft translation")
        print("  2. Compare with native Daft implementations")
        print("  3. Hand-optimize for better performance")
        print("=" * 70 + "\n")
        
        return 0
    
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


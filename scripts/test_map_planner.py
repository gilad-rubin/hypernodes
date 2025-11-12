"""Test script to verify MapPlanner refactoring works correctly."""

from hypernodes import Pipeline, node


@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input value."""
    return x * 2


@node(output_name="sum")
def add_two(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_zip_mode():
    """Test map operation in zip mode."""
    print("Testing zip mode...")
    pipeline = Pipeline(nodes=[add_two])
    
    results = pipeline.map(
        inputs={"a": [1, 2, 3], "b": [10, 20, 30]},
        map_over=["a", "b"],
        map_mode="zip"
    )
    
    print(f"  Input a: [1, 2, 3]")
    print(f"  Input b: [10, 20, 30]")
    print(f"  Results: {results}")
    
    # Expected: sum=[11, 22, 33]
    assert results["sum"] == [11, 22, 33], f"Expected [11, 22, 33], got {results['sum']}"
    print("  ✓ Zip mode passed!")


def test_product_mode():
    """Test map operation in product mode."""
    print("\nTesting product mode...")
    pipeline = Pipeline(nodes=[add_two])
    
    results = pipeline.map(
        inputs={"a": [1, 2], "b": [10, 20]},
        map_over=["a", "b"],
        map_mode="product"
    )
    
    print(f"  Input a: [1, 2]")
    print(f"  Input b: [10, 20]")
    print(f"  Results: {results}")
    
    # Expected combinations: (1,10)→11, (1,20)→21, (2,10)→12, (2,20)→22
    assert results["sum"] == [11, 21, 12, 22], f"Expected [11, 21, 12, 22], got {results['sum']}"
    print("  ✓ Product mode passed!")


def test_single_param():
    """Test map with single parameter."""
    print("\nTesting single parameter map...")
    pipeline = Pipeline(nodes=[double])
    
    results = pipeline.map(
        inputs={"x": [5, 10, 15]},
        map_over="x"
    )
    
    print(f"  Input x: [5, 10, 15]")
    print(f"  Results: {results}")
    
    assert results["doubled"] == [10, 20, 30], f"Expected [10, 20, 30], got {results['doubled']}"
    print("  ✓ Single parameter map passed!")


def test_with_fixed_params():
    """Test map with mixed varying and fixed parameters."""
    print("\nTesting with fixed parameters...")
    pipeline = Pipeline(nodes=[add_two])
    
    results = pipeline.map(
        inputs={"a": [1, 2, 3], "b": 100},  # b is fixed
        map_over=["a"],
        map_mode="zip"
    )
    
    print(f"  Input a (varying): [1, 2, 3]")
    print(f"  Input b (fixed): 100")
    print(f"  Results: {results}")
    
    # Expected: sum=[101, 102, 103]
    assert results["sum"] == [101, 102, 103], f"Expected [101, 102, 103], got {results['sum']}"
    print("  ✓ Fixed parameters test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MapPlanner Refactoring Test Suite")
    print("=" * 60)
    
    test_zip_mode()
    test_product_mode()
    test_single_param()
    test_with_fixed_params()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


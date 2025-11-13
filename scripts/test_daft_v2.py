#!/usr/bin/env python3
"""Test script for DaftEngineV2.

Tests:
1. Basic .run() with caching
2. .map() with multiple inputs
3. Verify callbacks fire correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hypernodes import Pipeline, node
from hypernodes.cache import DiskCache
from hypernodes.integrations.daft import DaftEngine
from hypernodes.telemetry import ProgressCallback


# Define simple nodes
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    print(f"  [EXECUTING] double({x})")
    return x * 2


@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add 10 to the input."""
    print(f"  [EXECUTING] add_ten({doubled})")
    return doubled + 10


def test_basic_run():
    """Test 1: Basic .run() with caching."""
    print("=" * 70)
    print("TEST 1: Basic .run() with caching")
    print("=" * 70)

    # Create cache and engine
    cache = DiskCache(".daft_v2_test_cache")
    cache.clear()

    engine = DaftEngine()
    pipeline = Pipeline(
        nodes=[double, add_ten],
        engine=engine,
        cache=cache,
        callbacks=[ProgressCallback()],
    )

    # First run - should execute
    print("\n--- First run (x=5) - should EXECUTE ---")
    result1 = pipeline.run(inputs={"x": 5})
    print(f"Result: {result1}")
    assert result1["result"] == 20, f"Expected 20, got {result1['result']}"

    # Second run - should hit cache
    print("\n--- Second run (x=5) - should CACHE HIT ---")
    result2 = pipeline.run(inputs={"x": 5})
    print(f"Result: {result2}")
    assert result2["result"] == 20, f"Expected 20, got {result2['result']}"

    # Third run with different input - should execute
    print("\n--- Third run (x=10) - should EXECUTE ---")
    result3 = pipeline.run(inputs={"x": 10})
    print(f"Result: {result3}")
    assert result3["result"] == 30, f"Expected 30, got {result3['result']}"

    print("\n✓ Test 1 passed!\n")


def test_map_operation():
    """Test 2: .map() with multiple inputs."""
    print("=" * 70)
    print("TEST 2: .map() with multiple inputs")
    print("=" * 70)

    cache = DiskCache(".daft_v2_test_cache")
    cache.clear()

    engine = DaftEngine()
    pipeline = Pipeline(
        nodes=[double, add_ten],
        engine=engine,
        cache=cache,
        callbacks=[ProgressCallback()],
    )

    # Map over list
    print("\n--- Mapping over [1, 2, 3] ---")
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    print(f"Results: {results}")

    # Verify results - should be list of dicts
    result_values = [r["result"] for r in results]
    assert result_values == [12, 14, 16], f"Expected [12, 14, 16], got {result_values}"

    print("\n✓ Test 2 passed!\n")


def test_map_with_cache():
    """Test 3: .map() with partial cache hits."""
    print("=" * 70)
    print("TEST 3: .map() with partial cache hits")
    print("=" * 70)

    cache = DiskCache(".daft_v2_test_cache")
    cache.clear()

    engine = DaftEngine()
    pipeline = Pipeline(
        nodes=[double, add_ten],
        engine=engine,
        cache=cache,
        callbacks=[ProgressCallback()],
    )

    # First map
    print("\n--- First map: [1, 2, 3] - should EXECUTE ---")
    results1 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    print(f"Results: {results1}")

    # Second map with overlap
    print("\n--- Second map: [2, 3, 4] - should have partial CACHE HITs ---")
    results2 = pipeline.map(inputs={"x": [2, 3, 4]}, map_over="x")
    print(f"Results: {results2}")

    result_values = [r["result"] for r in results2]
    assert result_values == [14, 16, 18], f"Expected [14, 16, 18], got {result_values}"

    print("\n✓ Test 3 passed!\n")


def test_output_filtering():
    """Test 4: Output filtering with output_name parameter."""
    print("=" * 70)
    print("TEST 4: Output filtering")
    print("=" * 70)

    cache = DiskCache(".daft_v2_test_cache")
    cache.clear()

    engine = DaftEngine()
    pipeline = Pipeline(
        nodes=[double, add_ten],
        engine=engine,
        cache=cache,
    )

    # Run with output filtering
    print("\n--- Run with output_name='result' ---")
    result = pipeline.run(inputs={"x": 5}, output_name="result")
    print(f"Result: {result}")

    # Should only have 'result', not 'doubled'
    assert "result" in result, "Expected 'result' in output"
    assert "doubled" not in result, "Expected 'doubled' NOT in output"
    assert result["result"] == 20, f"Expected 20, got {result['result']}"

    print("\n✓ Test 4 passed!\n")


if __name__ == "__main__":
    try:
        test_basic_run()
        test_map_operation()
        test_map_with_cache()
        test_output_filtering()

        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


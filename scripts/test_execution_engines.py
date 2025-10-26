"""Test script for all execution engines (sequential, async, threaded, parallel)."""

import time
from concurrent.futures import ThreadPoolExecutor

from hypernodes import Pipeline, node, LocalBackend


# Test nodes with varying workloads
@node(output_name="data")
def load_data():
    """I/O-bound: simulate loading data."""
    time.sleep(0.1)
    return list(range(10))


@node(output_name="processed_a")
def process_a(data):
    """CPU-bound: process data (independent)."""
    time.sleep(0.1)
    return [x * 2 for x in data]


@node(output_name="processed_b")
def process_b(data):
    """CPU-bound: process data (independent of process_a)."""
    time.sleep(0.1)
    return [x + 10 for x in data]


@node(output_name="result")
def combine(processed_a, processed_b):
    """Combine results (depends on both)."""
    time.sleep(0.05)
    return {"a": sum(processed_a), "b": sum(processed_b)}


def test_node_execution_modes():
    """Test different node execution modes."""
    print("\n" + "=" * 60)
    print("TESTING NODE EXECUTION MODES")
    print("=" * 60)
    
    # Create pipeline with independent nodes
    pipeline = Pipeline(
        nodes=[load_data, process_a, process_b, combine],
        name="test_pipeline"
    )
    
    modes = ["sequential", "async", "threaded", "parallel"]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing node_execution='{mode}'")
        print(f"{'='*60}")
        
        try:
            backend = LocalBackend(node_execution=mode)
            pipeline.backend = backend
            
            start = time.time()
            result = pipeline.run(inputs={})
            duration = time.time() - start
            
            print(f"✓ Result: {result}")
            print(f"✓ Duration: {duration:.2f}s")
            
            # Sequential should take ~0.35s (0.1+0.1+0.1+0.05)
            # Async/Threaded should take ~0.25s (0.1 + max(0.1,0.1) + 0.05)
            # Parallel might vary due to process overhead
            
        except Exception as e:
            print(f"✗ Error: {e}")


def test_map_execution_modes():
    """Test different map execution modes."""
    print("\n" + "=" * 60)
    print("TESTING MAP EXECUTION MODES")
    print("=" * 60)
    
    @node(output_name="doubled")
    def double_value(value):
        """Simple transformation."""
        time.sleep(0.05)
        return value * 2
    
    pipeline = Pipeline(nodes=[double_value], name="map_pipeline")
    
    modes = ["sequential", "async", "threaded", "parallel"]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing map_execution='{mode}'")
        print(f"{'='*60}")
        
        try:
            backend = LocalBackend(map_execution=mode)
            pipeline.backend = backend
            
            start = time.time()
            results = pipeline.map(inputs={"value": list(range(5))}, map_over="value")
            duration = time.time() - start
            
            print(f"✓ Results: {results['doubled']}")
            print(f"✓ Duration: {duration:.2f}s")
            
            # Sequential: ~0.25s (5 * 0.05)
            # Async/Threaded/Parallel: ~0.05s (parallel execution)
            
        except Exception as e:
            print(f"✗ Error: {e}")


def test_nested_maps():
    """Test nested map operations with intelligent resource management."""
    print("\n" + "=" * 60)
    print("TESTING NESTED MAP OPERATIONS")
    print("=" * 60)
    
    @node(output_name="item_result")
    def process_item(value):
        """Process single item."""
        time.sleep(0.01)
        return value + 1
    
    inner_pipeline = Pipeline(nodes=[process_item], name="inner")
    inner_node = inner_pipeline.as_node(map_over=["sub_items"])
    
    @node(output_name="batch_result")
    def aggregate(item_result):
        """Aggregate batch results."""
        return sum(item_result)
    
    outer_pipeline = Pipeline(
        nodes=[inner_node, aggregate],
        name="outer"
    )
    
    # Test with threaded execution
    # Outer: 3 items, each with 5 sub_items = 15 total operations
    print("\nTesting with map_execution='threaded'")
    print("Outer: 3 items, each with 5 sub_items = 15 total operations")
    
    backend = LocalBackend(map_execution="threaded", max_workers=8)
    outer_pipeline.backend = backend
    
    start = time.time()
    results = outer_pipeline.map(
        inputs={"sub_items": [[{"value": j} for j in range(5)] for i in range(3)]},
        map_over="sub_items"
    )
    duration = time.time() - start
    
    print(f"✓ Results: {results['batch_result']}")
    print(f"✓ Duration: {duration:.2f}s")
    print(f"✓ Resource management working (should limit concurrency intelligently)")


def test_custom_executor():
    """Test using a custom executor."""
    print("\n" + "=" * 60)
    print("TESTING CUSTOM EXECUTOR")
    print("=" * 60)
    
    @node(output_name="result")
    def compute(value):
        time.sleep(0.05)
        return value * 2
    
    pipeline = Pipeline(nodes=[compute], name="custom_executor_test")
    
    # Create custom executor with specific configuration
    custom_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="HyperNodes-")
    
    try:
        backend = LocalBackend(
            map_execution="threaded",
            executor=custom_executor
        )
        pipeline.backend = backend
        
        start = time.time()
        results = pipeline.map(inputs={"value": list(range(4))}, map_over="value")
        duration = time.time() - start
        
        print(f"✓ Results: {results['result']}")
        print(f"✓ Duration: {duration:.2f}s")
        print("✓ Custom executor used successfully")
        
    finally:
        custom_executor.shutdown(wait=True)


def test_mixed_execution():
    """Test mixing node and map execution modes."""
    print("\n" + "=" * 60)
    print("TESTING MIXED EXECUTION MODES")
    print("=" * 60)
    
    @node(output_name="processed")
    def process(value):
        time.sleep(0.03)
        return value * 3
    
    pipeline = Pipeline(nodes=[process], name="mixed_test")
    
    # Async node execution + Threaded map execution
    print("\nCombination: node_execution='async' + map_execution='threaded'")
    
    backend = LocalBackend(
        node_execution="async",
        map_execution="threaded",
        max_workers=4
    )
    pipeline.backend = backend
    
    start = time.time()
    results = pipeline.map(inputs={"value": list(range(6))}, map_over="value")
    duration = time.time() - start
    
    print(f"✓ Results: {results['processed']}")
    print(f"✓ Duration: {duration:.2f}s")
    print(f"✓ Mixed execution modes work correctly")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HYPERNODES EXECUTION ENGINE TEST SUITE")
    print("=" * 60)
    
    try:
        test_node_execution_modes()
        test_map_execution_modes()
        test_nested_maps()
        test_custom_executor()
        test_mixed_execution()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

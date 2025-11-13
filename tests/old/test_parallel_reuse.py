"""Test that parallel executors can be reused without shutdown errors."""

import time

from hypernodes import HypernodesEngine, Pipeline, node


def test_parallel_executor_reuse():
    """Test that loky's reusable executor can be used multiple times."""
    
    @node(output_name="result")
    def compute(x: int) -> int:
        return x ** 2
    
    # Create and use first pipeline
    pipeline1 = Pipeline(
        nodes=[compute],
        engine=HypernodesEngine(map_executor="parallel", max_workers=4)
    )
    results1 = pipeline1.map(inputs={"x": [1, 2, 3, 4]}, map_over="x")
    assert results1["result"] == [1, 4, 9, 16]
    
    # Create and use second pipeline (should reuse same executor)
    pipeline2 = Pipeline(
        nodes=[compute],
        engine=HypernodesEngine(map_executor="parallel", max_workers=4)
    )
    results2 = pipeline2.map(inputs={"x": [5, 6, 7, 8]}, map_over="x")
    assert results2["result"] == [25, 36, 49, 64]
    
    # Explicitly cleanup first pipeline
    del pipeline1
    
    # Should still work with second pipeline
    results3 = pipeline2.map(inputs={"x": [9, 10]}, map_over="x")
    assert results3["result"] == [81, 100]
    
    # Create third pipeline after first is deleted
    pipeline3 = Pipeline(
        nodes=[compute],
        engine=HypernodesEngine(map_executor="parallel", max_workers=4)
    )
    results4 = pipeline3.map(inputs={"x": [11, 12]}, map_over="x")
    assert results4["result"] == [121, 144]
    
    print("✅ Parallel executor reuse test passed!")


def test_multiple_sequential_parallel_pipelines():
    """Test creating and running multiple parallel pipelines in sequence."""
    
    @node(output_name="result")
    def process(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    # Run 5 pipelines sequentially
    for i in range(5):
        pipeline = Pipeline(
            nodes=[process],
            engine=HypernodesEngine(map_executor="parallel", max_workers=2)
        )
        results = pipeline.map(inputs={"x": [i, i+1]}, map_over="x")
        expected = [i*2, (i+1)*2]
        assert results["result"] == expected, f"Iteration {i} failed"
    
    print("✅ Sequential parallel pipelines test passed!")


if __name__ == "__main__":
    test_parallel_executor_reuse()
    test_multiple_sequential_parallel_pipelines()

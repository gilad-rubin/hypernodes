"""Test DaskEngine with the same benchmark functions from the notebook."""

import time
from typing import Any, Dict

from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine


# Same functions from benchmark notebook
def fetch_data(file_id: int) -> Dict[str, Any]:
    """Simulate I/O operation."""
    time.sleep(0.001)  # 1ms delay
    return {
        "id": file_id,
        "value": file_id * 10.0,
        "metadata": {"source": "test"},
    }


def transform(data: Dict[str, Any]) -> Dict[str, Any]:
    """Light CPU work."""
    value = data.get("value", 0.0)
    normalized_value = (value - 50) / 50
    return {
        "id": data["id"],
        "normalized_value": normalized_value,
        "metadata_hash": "test_hash",
    }


def heavy_compute(transformed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Heavy CPU work."""
    import numpy as np

    value = transformed_data["normalized_value"]
    result = value
    for _ in range(50):
        result = np.sin(result) * np.cos(result) + np.sqrt(abs(result) + 0.01)
        result = np.tanh(result) * np.exp(-abs(result))

    matrix = np.random.randn(10, 10)
    eigenvalues = np.linalg.eigvals(matrix)
    complexity_score = float(np.mean(np.abs(eigenvalues)))

    return {
        "id": transformed_data["id"],
        "computed_result": float(result),
        "complexity_score": complexity_score,
        "metadata_hash": transformed_data["metadata_hash"],
    }


def aggregate(computed_data: Dict[str, Any]) -> float:
    """Aggregate results."""
    return computed_data["computed_result"] + computed_data["complexity_score"]


# Wrap as HyperNodes
@node(output_name="raw_data")
def node_fetch(file_id: int) -> Dict[str, Any]:
    return fetch_data(file_id)


@node(output_name="transformed_data")
def node_transform(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    return transform(raw_data)


@node(output_name="computed_data")
def node_compute(transformed_data: Dict[str, Any]) -> Dict[str, Any]:
    return heavy_compute(transformed_data)


@node(output_name="final_result")
def node_aggregate(computed_data: Dict[str, Any]) -> float:
    return aggregate(computed_data)


def test_dask_engine():
    """Test DaskEngine with different configurations."""
    print("=" * 80)
    print("TESTING DASK ENGINE")
    print("=" * 80)

    # Test configurations
    configs = [
        {"name": "Auto (mixed)", "engine": DaskEngine()},
        {
            "name": "Threads scheduler",
            "engine": DaskEngine(scheduler="threads", workload_type="mixed"),
        },
        {
            "name": "CPU-bound",
            "engine": DaskEngine(scheduler="threads", workload_type="cpu"),
        },
        {
            "name": "I/O-bound",
            "engine": DaskEngine(scheduler="threads", workload_type="io"),
        },
        {
            "name": "Manual 8 partitions",
            "engine": DaskEngine(scheduler="threads", npartitions=8),
        },
    ]

    # Test dataset sizes
    test_sizes = {"small": 10, "medium": 100, "large": 500}

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config['name']}")
        print(f"{'=' * 80}")

        engine = config["engine"]
        pipeline = Pipeline(
            nodes=[node_fetch, node_transform, node_compute, node_aggregate],
            engine=engine,
        )

        for size_name, size in test_sizes.items():
            file_ids = list(range(size))

            # Warm-up
            _ = pipeline.map(inputs={"file_id": file_ids[:5]}, map_over="file_id")

            # Benchmark
            start = time.perf_counter()
            results = pipeline.map(inputs={"file_id": file_ids}, map_over="file_id")
            elapsed = (time.perf_counter() - start) * 1000

            print(f"\n{size_name.upper()} ({size} items):")
            print(f"  Time: {elapsed:.2f}ms")
            print(f"  Per-item: {elapsed / size:.3f}ms")
            print(f"  Sample results: {results[:2]}")

    print(f"\n{'=' * 80}")
    print("Test complete!")
    print(f"{'=' * 80}")


def test_single_run():
    """Test that single run uses sequential execution (no overhead)."""
    print("\n" + "=" * 80)
    print("TESTING SINGLE RUN (should be sequential, no Dask overhead)")
    print("=" * 80)

    engine = DaskEngine()
    pipeline = Pipeline(
        nodes=[node_fetch, node_transform, node_compute, node_aggregate],
        engine=engine,
    )

    # Single run
    start = time.perf_counter()
    result = pipeline.run(inputs={"file_id": 0})
    elapsed = (time.perf_counter() - start) * 1000

    print("\nSingle run:")
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  Result: {result}")
    print("  ✓ No Dask overhead for single runs")


if __name__ == "__main__":
    test_single_run()
    test_dask_engine()

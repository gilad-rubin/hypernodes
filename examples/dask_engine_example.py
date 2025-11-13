"""Simple example demonstrating DaskEngine usage.

This example shows how easy it is to get parallel execution with HyperNodes
using the DaskEngine - just one line of configuration!
"""

import time

from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine, SequentialEngine
from hypernodes.telemetry import ProgressCallback


# Define a simple CPU-intensive task
@node(output_name="result")
def fibonacci(n: int) -> int:
    """Calculate Fibonacci number (inefficiently for demonstration)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@node(output_name="squared")
def square(result: int) -> int:
    """Square the result."""
    return result * result


@node(output_name="message")
def format_message(n: int, squared: int) -> str:
    """Create a message."""
    return f"fib({n})^2 = {squared}"


def main():
    print("=" * 80)
    print("HYPERNODES DASKENGINE EXAMPLE")
    print("=" * 80)

    # Test data
    numbers = list(range(10, 21))  # Calculate fib(10) through fib(20)

    print(f"\nCalculating Fibonacci numbers for: {numbers}")
    print(f"Number of items: {len(numbers)}\n")

    # 1. Sequential Engine (baseline)
    print("-" * 80)
    print("1. SEQUENTIAL ENGINE (baseline)")
    print("-" * 80)

    sequential_pipeline = Pipeline(
        nodes=[fibonacci, square, format_message],
        engine=SequentialEngine(),
    )

    start = time.perf_counter()
    sequential_results = sequential_pipeline.map(
        inputs={"n": numbers}, map_over="n"
    )
    sequential_time = (time.perf_counter() - start) * 1000

    print(f"Time: {sequential_time:.2f}ms")
    print(f"Results: {sequential_results[:3]}")

    # 2. DaskEngine (auto-optimized)
    print("\n" + "-" * 80)
    print("2. DASKENGINE (auto-optimized, zero configuration)")
    print("-" * 80)

    dask_pipeline = Pipeline(
        nodes=[fibonacci, square, format_message],
        engine=DaskEngine(),  # That's it! One line for parallelism
        callbacks=[ProgressCallback()],  # Optional: show progress
    )

    start = time.perf_counter()
    dask_results = dask_pipeline.map(inputs={"n": numbers}, map_over="n")
    dask_time = (time.perf_counter() - start) * 1000

    print(f"Time: {dask_time:.2f}ms")
    print(f"Results: {dask_results[:3]}")

    # 3. DaskEngine (CPU-optimized)
    print("\n" + "-" * 80)
    print("3. DASKENGINE (CPU-optimized configuration)")
    print("-" * 80)

    cpu_optimized_pipeline = Pipeline(
        nodes=[fibonacci, square, format_message],
        engine=DaskEngine(
            scheduler="threads",
            workload_type="cpu",  # Optimize for CPU-bound work
        ),
    )

    start = time.perf_counter()
    cpu_results = cpu_optimized_pipeline.map(inputs={"n": numbers}, map_over="n")
    cpu_time = (time.perf_counter() - start) * 1000

    print(f"Time: {cpu_time:.2f}ms")
    print(f"Results: {cpu_results[:3]}")

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Sequential Engine:      {sequential_time:.2f}ms")
    print(f"DaskEngine (auto):      {dask_time:.2f}ms (speedup: {sequential_time / dask_time:.2f}x)")
    print(f"DaskEngine (CPU-opt):   {cpu_time:.2f}ms (speedup: {sequential_time / cpu_time:.2f}x)")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("✓ DaskEngine provides automatic parallelism with ONE line of code")
    print("✓ No Dask knowledge required - sensible defaults work great")
    print("✓ Can fine-tune for specific workloads (CPU, I/O, mixed)")
    print("✓ Significant speedup for map operations")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Performance tests for different executor types.

These tests verify that async, threaded, and parallel executors
actually improve performance over sequential execution for appropriate workloads.
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine


# ============================================================================
# Async Executor Tests (I/O-bound work)
# ============================================================================


@node(output_name="fetched")
async def async_fetch_data(url: str) -> dict:
    """Simulate async I/O-bound API call."""
    await asyncio.sleep(0.1)  # Simulate network latency
    return {"url": url, "data": f"content_from_{url}"}


@node(output_name="parsed")
async def async_parse_data(fetched: dict) -> str:
    """Simulate async parsing."""
    await asyncio.sleep(0.05)  # Simulate parsing time
    return fetched["data"].upper()


def test_async_executor_map_performance():
    """Verify async executor is faster than sequential for I/O-bound map operations."""
    # Sequential execution
    pipeline_seq = Pipeline(
        nodes=[async_fetch_data, async_parse_data],
        backend=HypernodesEngine(
            node_executor="sequential",
            map_executor="sequential",
        ),
    )

    urls = [f"http://api{i}.com" for i in range(10)]

    start = time.time()
    result_seq = pipeline_seq.map(
        inputs={"urls": urls},
        map_over="urls",
    )
    time_seq = time.time() - start

    # Async execution
    pipeline_async = Pipeline(
        nodes=[async_fetch_data, async_parse_data],
        backend=HypernodesEngine(
            node_executor="sequential",  # Nodes run in sequence (fetch then parse)
            map_executor="async",  # But map items run concurrently
        ),
    )

    start = time.time()
    result_async = pipeline_async.map(
        inputs={"urls": urls},
        map_over="urls",
    )
    time_async = time.time() - start

    # Verify results are identical
    assert result_seq["parsed"] == result_async["parsed"]

    # Async should be significantly faster (at least 3x for 10 items)
    print(f"\nAsync I/O Map Performance:")
    print(f"  Sequential: {time_seq:.2f}s")
    print(f"  Async:      {time_async:.2f}s")
    print(f"  Speedup:    {time_seq / time_async:.1f}x")

    assert time_async < time_seq / 3, f"Async should be at least 3x faster, but was only {time_seq / time_async:.1f}x"


def test_async_executor_node_performance():
    """Verify async executor is faster than sequential for independent I/O-bound nodes."""

    @node(output_name="api1_data")
    async def fetch_api1() -> str:
        await asyncio.sleep(0.1)
        return "data1"

    @node(output_name="api2_data")
    async def fetch_api2() -> str:
        await asyncio.sleep(0.1)
        return "data2"

    @node(output_name="api3_data")
    async def fetch_api3() -> str:
        await asyncio.sleep(0.1)
        return "data3"

    @node(output_name="combined")
    def combine(api1_data: str, api2_data: str, api3_data: str) -> str:
        return f"{api1_data},{api2_data},{api3_data}"

    # Sequential execution
    pipeline_seq = Pipeline(
        nodes=[fetch_api1, fetch_api2, fetch_api3, combine],
        backend=HypernodesEngine(node_executor="sequential"),
    )

    start = time.time()
    result_seq = pipeline_seq.run(inputs={})
    time_seq = time.time() - start

    # Async execution - independent nodes run concurrently
    pipeline_async = Pipeline(
        nodes=[fetch_api1, fetch_api2, fetch_api3, combine],
        backend=HypernodesEngine(node_executor="async"),
    )

    start = time.time()
    result_async = pipeline_async.run(inputs={})
    time_async = time.time() - start

    # Verify results are identical
    assert result_seq["combined"] == result_async["combined"] == "data1,data2,data3"

    # Async should be significantly faster (close to 3x for 3 independent nodes)
    print(f"\nAsync I/O Node Performance:")
    print(f"  Sequential: {time_seq:.2f}s")
    print(f"  Async:      {time_async:.2f}s")
    print(f"  Speedup:    {time_seq / time_async:.1f}x")

    assert time_async < time_seq / 2, f"Async should be at least 2x faster, but was only {time_seq / time_async:.1f}x"


# ============================================================================
# Threaded Executor Tests (I/O-bound work with blocking calls)
# ============================================================================


@node(output_name="file_data")
def blocking_read_file(filename: str) -> str:
    """Simulate blocking I/O (file read, database query, etc)."""
    time.sleep(0.1)  # Simulate blocking I/O
    return f"contents_of_{filename}"


@node(output_name="processed")
def process_file(file_data: str) -> str:
    """Simulate some processing."""
    time.sleep(0.05)
    return file_data.upper()


def test_threaded_executor_map_performance():
    """Verify threaded executor is faster than sequential for blocking I/O map operations."""
    # Sequential execution
    pipeline_seq = Pipeline(
        nodes=[blocking_read_file, process_file],
        backend=HypernodesEngine(
            node_executor="sequential",
            map_executor="sequential",
        ),
    )

    filenames = [f"file_{i}.txt" for i in range(8)]

    start = time.time()
    result_seq = pipeline_seq.map(
        inputs={"filenames": filenames},
        map_over="filenames",
    )
    time_seq = time.time() - start

    # Threaded execution
    pipeline_threaded = Pipeline(
        nodes=[blocking_read_file, process_file],
        backend=HypernodesEngine(
            node_executor="sequential",
            map_executor=ThreadPoolExecutor(max_workers=4),
        ),
    )

    start = time.time()
    result_threaded = pipeline_threaded.map(
        inputs={"filenames": filenames},
        map_over="filenames",
    )
    time_threaded = time.time() - start

    # Verify results are identical
    assert result_seq["processed"] == result_threaded["processed"]

    # Threaded should be significantly faster (close to 4x with 4 workers)
    print(f"\nThreaded I/O Map Performance:")
    print(f"  Sequential: {time_seq:.2f}s")
    print(f"  Threaded:   {time_threaded:.2f}s")
    print(f"  Speedup:    {time_seq / time_threaded:.1f}x")

    assert time_threaded < time_seq / 2, f"Threaded should be at least 2x faster, but was only {time_seq / time_threaded:.1f}x"


# ============================================================================
# Parallel Executor Tests (CPU-bound work)
# ============================================================================


@node(output_name="computed")
def cpu_intensive(n: int) -> int:
    """Simulate CPU-intensive computation."""
    # Calculate sum of squares (CPU-bound)
    result = 0
    for i in range(n):
        result += i * i
    return result


@node(output_name="doubled")
def double_result(computed: int) -> int:
    """Another CPU operation."""
    return computed * 2


def test_parallel_executor_map_performance():
    """Verify parallel executor is faster than sequential for CPU-bound map operations."""
    # Sequential execution
    pipeline_seq = Pipeline(
        nodes=[cpu_intensive, double_result],
        backend=HypernodesEngine(
            node_executor="sequential",
            map_executor="sequential",
        ),
    )

    # Use moderate computation to see clear speedup
    numbers = [500_000] * 8

    start = time.time()
    result_seq = pipeline_seq.map(
        inputs={"numbers": numbers},
        map_over="numbers",
    )
    time_seq = time.time() - start

    # Parallel execution
    pipeline_parallel = Pipeline(
        nodes=[cpu_intensive, double_result],
        backend=HypernodesEngine(
            node_executor="sequential",
            map_executor=ProcessPoolExecutor(max_workers=4),
        ),
    )

    start = time.time()
    result_parallel = pipeline_parallel.map(
        inputs={"numbers": numbers},
        map_over="numbers",
    )
    time_parallel = time.time() - start

    # Verify results are identical
    assert result_seq["doubled"] == result_parallel["doubled"]

    # Parallel should be faster (at least 1.5x with overhead)
    print(f"\nParallel CPU Map Performance:")
    print(f"  Sequential: {time_seq:.2f}s")
    print(f"  Parallel:   {time_parallel:.2f}s")
    print(f"  Speedup:    {time_seq / time_parallel:.1f}x")

    assert time_parallel < time_seq / 1.5, f"Parallel should be at least 1.5x faster, but was only {time_seq / time_parallel:.1f}x"


def test_parallel_executor_node_performance():
    """Verify parallel executor can speed up independent CPU-bound nodes."""

    @node(output_name="result1")
    def compute1(base: int) -> int:
        result = 0
        for i in range(base):
            result += i * i
        return result

    @node(output_name="result2")
    def compute2(base: int) -> int:
        result = 0
        for i in range(base):
            result += i * i * i
        return result

    @node(output_name="total")
    def combine(result1: int, result2: int) -> int:
        return result1 + result2

    # Sequential execution
    pipeline_seq = Pipeline(
        nodes=[compute1, compute2, combine],
        backend=HypernodesEngine(node_executor="sequential"),
    )

    start = time.time()
    result_seq = pipeline_seq.run(inputs={"base": 1_000_000})
    time_seq = time.time() - start

    # Parallel execution - independent nodes run in parallel processes
    pipeline_parallel = Pipeline(
        nodes=[compute1, compute2, combine],
        backend=HypernodesEngine(node_executor=ProcessPoolExecutor(max_workers=2)),
    )

    start = time.time()
    result_parallel = pipeline_parallel.run(inputs={"base": 1_000_000})
    time_parallel = time.time() - start

    # Verify results are identical
    assert result_seq["total"] == result_parallel["total"]

    # Parallel should be faster (at least 1.3x accounting for overhead)
    print(f"\nParallel CPU Node Performance:")
    print(f"  Sequential: {time_seq:.2f}s")
    print(f"  Parallel:   {time_parallel:.2f}s")
    print(f"  Speedup:    {time_seq / time_parallel:.1f}x")

    assert time_parallel < time_seq / 1.3, f"Parallel should be at least 1.3x faster, but was only {time_seq / time_parallel:.1f}x"


# ============================================================================
# Mixed Executor Test
# ============================================================================


def test_mixed_executors():
    """Verify we can use different executors for nodes vs map operations."""

    @node(output_name="fetched")
    async def fetch_url(url: str) -> str:
        await asyncio.sleep(0.05)
        return f"data_from_{url}"

    @node(output_name="computed")
    def heavy_compute(fetched: str, n: int) -> int:
        # CPU-intensive
        result = 0
        for i in range(n):
            result += i
        return len(fetched) + result

    # Use async for nodes (fetch can be async) and parallel for map (CPU compute)
    pipeline = Pipeline(
        nodes=[fetch_url, heavy_compute],
        backend=HypernodesEngine(
            node_executor="async",  # Async for the async fetch node
            map_executor=ProcessPoolExecutor(max_workers=2),  # Parallel for map items
        ),
    )

    start = time.time()
    result = pipeline.map(
        inputs={
            "urls": ["http://a.com", "http://b.com"],
            "n": [300_000, 300_000],
        },
        map_over=["urls", "n"],
    )
    time_elapsed = time.time() - start

    print(f"\nMixed Executors Performance: {time_elapsed:.2f}s")

    # Should complete successfully with correct results
    assert len(result["computed"]) == 2
    assert all(isinstance(x, int) for x in result["computed"])


# ============================================================================
# String Alias Tests
# ============================================================================


def test_string_aliases():
    """Verify string aliases work correctly."""

    @node(output_name="result")
    async def async_work(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    # Test "sequential" alias
    pipeline_seq = Pipeline(
        nodes=[async_work],
        backend=HypernodesEngine(
            node_executor="sequential",
            map_executor="sequential",
        ),
    )
    result_seq = pipeline_seq.run(inputs={"x": 5})
    assert result_seq["result"] == 10

    # Test "async" alias
    pipeline_async = Pipeline(
        nodes=[async_work],
        backend=HypernodesEngine(
            node_executor="async",
            map_executor="async",
        ),
    )
    result_async = pipeline_async.run(inputs={"x": 5})
    assert result_async["result"] == 10

    # Test "threaded" alias
    @node(output_name="sync_result")
    def sync_work(x: int) -> int:
        time.sleep(0.01)
        return x * 3

    pipeline_threaded = Pipeline(
        nodes=[sync_work],
        backend=HypernodesEngine(
            node_executor="threaded",
            map_executor="threaded",
        ),
    )
    result_threaded = pipeline_threaded.run(inputs={"x": 5})
    assert result_threaded["sync_result"] == 15

    # Test "parallel" alias
    pipeline_parallel = Pipeline(
        nodes=[sync_work],
        backend=HypernodesEngine(
            node_executor="parallel",
            map_executor="parallel",
        ),
    )
    result_parallel = pipeline_parallel.run(inputs={"x": 5})
    assert result_parallel["sync_result"] == 15

    print("\n✓ All string aliases work correctly")


if __name__ == "__main__":
    # Run tests manually to see performance output
    print("=" * 70)
    print("EXECUTOR PERFORMANCE TESTS")
    print("=" * 70)

    test_async_executor_map_performance()
    test_async_executor_node_performance()
    test_threaded_executor_map_performance()
    test_parallel_executor_map_performance()
    test_parallel_executor_node_performance()
    test_mixed_executors()
    test_string_aliases()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)

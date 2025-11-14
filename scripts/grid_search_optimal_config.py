#!/usr/bin/env python3
"""
Grid Search for Optimal ThreadPoolExecutor Configuration

Based on research and DaskEngine heuristics:
- DaskEngine uses: I/O=4x cores, CPU=2x cores, Mixed=3x cores
- ThreadPoolExecutor best practices: I/O can use 5-10x cores
- Batch size: Balance overhead vs memory

This script tests different configurations to find optimal heuristics.
"""

import asyncio
import multiprocessing
import time
from typing import List

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine, DaskEngine

# ==================== Configuration Space ====================

CPU_COUNT = multiprocessing.cpu_count()
print(f"CPU Count: {CPU_COUNT}")

# Grid of max_workers to test (as multipliers of CPU count)
WORKER_MULTIPLIERS = [1, 2, 3, 4, 5, 8, 10, 16]

# Grid of batch sizes to test
BATCH_SIZES = [32, 64, 128, 256, 512, 1024, None]  # None = auto

# Test scales
TEST_SCALES = [50, 100, 200]


# ==================== Test Functions ====================


@node(output_name="result")
def sync_io_task(text: str, delay_ms: float = 10) -> str:
    """Sync I/O-bound task."""
    time.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


@node(output_name="result")
async def async_io_task(text: str, delay_ms: float = 10) -> str:
    """Async I/O-bound task."""
    await asyncio.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


# ==================== Test Data ====================


def generate_test_texts(count: int) -> List[str]:
    """Generate test text data."""
    return [f"Text item {i} for processing" for i in range(count)]


# ==================== Grid Search Functions ====================


def test_daft_sync_batch(
    texts: List[str], max_workers: int, batch_size: int = None, delay_ms: float = 10
) -> float:
    """Test DaftEngine with sync batch UDF."""
    config = {"max_workers": max_workers}
    if batch_size is not None:
        config["batch_size"] = batch_size

    pipeline = Pipeline(
        nodes=[sync_io_task],
        engine=DaftEngine(use_batch_udf=True, default_daft_config=config),
    )

    start = time.perf_counter()
    _ = pipeline.map(inputs={"text": texts, "delay_ms": delay_ms}, map_over="text")
    elapsed = time.perf_counter() - start

    return elapsed


def test_daft_async(texts: List[str], delay_ms: float = 10) -> float:
    """Test DaftEngine with async (no config needed)."""
    pipeline = Pipeline(nodes=[async_io_task], engine=DaftEngine())

    start = time.perf_counter()
    _ = pipeline.map(inputs={"text": texts, "delay_ms": delay_ms}, map_over="text")
    elapsed = time.perf_counter() - start

    return elapsed


def test_dask_threads(
    texts: List[str], npartitions: int = None, delay_ms: float = 10
) -> float:
    """Test DaskEngine with threads."""
    if npartitions is None:
        # Use DaskEngine's auto heuristic (4x for I/O)
        engine = DaskEngine(scheduler="threads", workload_type="io")
    else:
        engine = DaskEngine(scheduler="threads", npartitions=npartitions)

    pipeline = Pipeline(nodes=[sync_io_task], engine=engine)

    start = time.perf_counter()
    _ = pipeline.map(inputs={"text": texts, "delay_ms": delay_ms}, map_over="text")
    elapsed = time.perf_counter() - start

    return elapsed


# ==================== Grid Search: Max Workers ====================


def grid_search_max_workers(scale: int, delay_ms: float = 10):
    """Test different max_workers values."""
    print(f"\n{'=' * 80}")
    print(f"GRID SEARCH: max_workers (scale={scale} items)")
    print(f"{'=' * 80}")

    texts = generate_test_texts(scale)
    results = []

    # Baseline: sequential
    baseline_time = scale * (delay_ms / 1000)

    print(
        f"\n{'Multiplier':<12} {'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Status'}"
    )
    print(f"{'-' * 80}")

    for multiplier in WORKER_MULTIPLIERS:
        max_workers = multiplier * CPU_COUNT

        try:
            elapsed = test_daft_sync_batch(texts, max_workers, delay_ms=delay_ms)
            speedup = baseline_time / elapsed
            status = "✓"

            results.append(
                {
                    "multiplier": multiplier,
                    "max_workers": max_workers,
                    "elapsed": elapsed,
                    "speedup": speedup,
                }
            )

            print(
                f"{multiplier:<12} {max_workers:<10} {elapsed:<12.3f} {speedup:<10.2f}x {status}"
            )

        except Exception as e:
            print(f"{multiplier:<12} {max_workers:<10} FAILED: {str(e)[:30]}")

    # Find best
    best = max(results, key=lambda x: x["speedup"])
    print(
        f"\n🏆 Best: {best['multiplier']}x cores ({best['max_workers']} workers) → {best['speedup']:.2f}x speedup"
    )

    # Compare to DaskEngine heuristic (4x)
    dask_heuristic = 4
    dask_workers = dask_heuristic * CPU_COUNT
    dask_result = [r for r in results if r["multiplier"] == dask_heuristic]
    if dask_result:
        print(f"📊 DaskEngine heuristic (4x): {dask_result[0]['speedup']:.2f}x speedup")

    return results, best


# ==================== Grid Search: Batch Size ====================


def grid_search_batch_size(scale: int, max_workers: int, delay_ms: float = 10):
    """Test different batch_size values."""
    print(f"\n{'=' * 80}")
    print(f"GRID SEARCH: batch_size (scale={scale}, workers={max_workers})")
    print(f"{'=' * 80}")

    texts = generate_test_texts(scale)
    results = []
    baseline_time = scale * (delay_ms / 1000)

    print(f"\n{'Batch Size':<12} {'Time (s)':<12} {'Speedup':<10} {'Status'}")
    print(f"{'-' * 80}")

    for batch_size in BATCH_SIZES:
        try:
            elapsed = test_daft_sync_batch(texts, max_workers, batch_size, delay_ms)
            speedup = baseline_time / elapsed
            status = "✓"

            results.append(
                {
                    "batch_size": batch_size if batch_size else "auto",
                    "elapsed": elapsed,
                    "speedup": speedup,
                }
            )

            bs_str = str(batch_size) if batch_size else "auto"
            print(f"{bs_str:<12} {elapsed:<12.3f} {speedup:<10.2f}x {status}")

        except Exception as e:
            bs_str = str(batch_size) if batch_size else "auto"
            print(f"{bs_str:<12} FAILED: {str(e)[:40]}")

    # Find best
    best = max(results, key=lambda x: x["speedup"])
    print(
        f"\n🏆 Best: batch_size={best['batch_size']} → {best['speedup']:.2f}x speedup"
    )

    return results, best


# ==================== Direct Comparison: Dask vs Daft ====================


def compare_dask_vs_daft(scale: int, delay_ms: float = 10):
    """Direct comparison between DaskEngine and DaftEngine."""
    print(f"\n{'=' * 80}")
    print(f"DIRECT COMPARISON: DaskEngine vs DaftEngine (scale={scale})")
    print(f"{'=' * 80}")

    texts = generate_test_texts(scale)
    baseline_time = scale * (delay_ms / 1000)

    results = []

    print(f"\n{'Strategy':<50} {'Time (s)':<12} {'Speedup':<10}")
    print(f"{'-' * 80}")

    # 1. DaskEngine with auto heuristic (threads, I/O workload)
    elapsed = test_dask_threads(texts, npartitions=None, delay_ms=delay_ms)
    speedup = baseline_time / elapsed
    results.append(("DaskEngine (auto, threads, I/O)", elapsed, speedup))
    print(f"{'DaskEngine (auto, threads, I/O)':<50} {elapsed:<12.3f} {speedup:<10.2f}x")

    # 2. DaskEngine with 4x heuristic (explicit)
    nparts = 4 * CPU_COUNT
    elapsed = test_dask_threads(texts, npartitions=nparts, delay_ms=delay_ms)
    speedup = baseline_time / elapsed
    results.append((f"DaskEngine (4x cores = {nparts} partitions)", elapsed, speedup))
    print(
        f"{f'DaskEngine (4x cores = {nparts} partitions)':<50} {elapsed:<12.3f} {speedup:<10.2f}x"
    )

    # 3. DaftEngine sync batch with 4x workers (match Dask)
    workers_4x = 4 * CPU_COUNT
    elapsed = test_daft_sync_batch(texts, max_workers=workers_4x, delay_ms=delay_ms)
    speedup = baseline_time / elapsed
    results.append(
        (f"DaftEngine (sync batch, 4x cores = {workers_4x} workers)", elapsed, speedup)
    )
    print(
        f"{f'DaftEngine (sync batch, 4x cores = {workers_4x} workers)':<50} {elapsed:<12.3f} {speedup:<10.2f}x"
    )

    # 4. DaftEngine sync batch with 8x workers
    workers_8x = 8 * CPU_COUNT
    elapsed = test_daft_sync_batch(texts, max_workers=workers_8x, delay_ms=delay_ms)
    speedup = baseline_time / elapsed
    results.append(
        (f"DaftEngine (sync batch, 8x cores = {workers_8x} workers)", elapsed, speedup)
    )
    print(
        f"{f'DaftEngine (sync batch, 8x cores = {workers_8x} workers)':<50} {elapsed:<12.3f} {speedup:<10.2f}x"
    )

    # 5. DaftEngine async
    elapsed = test_daft_async(texts, delay_ms=delay_ms)
    speedup = baseline_time / elapsed
    results.append(("DaftEngine (async)", elapsed, speedup))
    print(f"{'DaftEngine (async)':<50} {elapsed:<12.3f} {speedup:<10.2f}x")

    # Find best
    best_name, best_time, best_speedup = max(results, key=lambda x: x[2])
    print(f"\n🏆 Winner: {best_name}")
    print(f"   Speedup: {best_speedup:.2f}x")

    # Comparison summary
    dask_auto = results[0]
    daft_async = results[4]
    print("\n📊 Key Comparison:")
    print(f"   DaskEngine (auto):  {dask_auto[2]:.2f}x speedup")
    print(f"   DaftEngine (async): {daft_async[2]:.2f}x speedup")
    print(f"   Improvement: {daft_async[2] / dask_auto[2]:.2f}x faster")

    return results


# ==================== Main ====================


def main():
    """Run all grid searches and comparisons."""
    print("\n" + "=" * 80)
    print("GRID SEARCH: OPTIMAL CONFIGURATION FOR PARALLELISM")
    print("=" * 80)
    print(f"\nSystem: {CPU_COUNT} CPU cores")

    # Test at medium scale (100 items)
    scale = 100
    delay_ms = 10

    # 1. Grid search: max_workers
    worker_results, best_workers = grid_search_max_workers(scale, delay_ms)

    # 2. Grid search: batch_size (using best max_workers)
    batch_results, best_batch = grid_search_batch_size(
        scale, best_workers["max_workers"], delay_ms
    )

    # 3. Direct comparison at multiple scales
    for test_scale in [50, 100, 200]:
        compare_dask_vs_daft(test_scale, delay_ms)

    # 4. Summary of findings
    print(f"\n{'=' * 80}")
    print("HEURISTICS SUMMARY")
    print(f"{'=' * 80}")

    print("\n📊 RECOMMENDED HEURISTICS:")
    print("\nFor ThreadPoolExecutor (I/O-bound tasks):")
    print(f"  Best multiplier: {best_workers['multiplier']}x CPU cores")
    print(f"  → For {CPU_COUNT} cores: {best_workers['max_workers']} workers")
    print(f"  → Speedup: {best_workers['speedup']:.2f}x")

    print("\nFor Batch Size:")
    print(f"  Best batch size: {best_batch['batch_size']}")
    print(f"  → Speedup: {best_batch['speedup']:.2f}x")

    print("\n📖 COMPARISON TO DASKENGINE HEURISTIC:")
    print("  DaskEngine uses: 4x CPU cores for I/O workload")
    print(f"  Our best: {best_workers['multiplier']}x CPU cores")

    if best_workers["multiplier"] > 4:
        print("  ✅ ThreadPoolExecutor can benefit from MORE workers than Dask!")
    elif best_workers["multiplier"] == 4:
        print("  ✅ DaskEngine heuristic is optimal!")
    else:
        print("  ⚠️  Fewer workers than Dask is better (less overhead)")

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print("\nFor DaftEngine with ThreadPoolExecutor:")
    print(f"  max_workers = {best_workers['multiplier']} × CPU_COUNT")
    print(f"  batch_size  = {best_batch['batch_size']}")
    print("\nExpected performance:")
    print(f"  I/O-bound sync: ~{best_workers['speedup']:.1f}x speedup")
    print("  I/O-bound async: ~30-37x speedup (Daft's native async)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Improved DaftEngine

Tests the fixes:
1. Async function support (37x speedup expected)
2. ThreadPoolExecutor in batch UDF (10x speedup expected)
"""

import asyncio
import time
from typing import List

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine, DaskEngine


# ==================== Test Functions ====================

@node(output_name="result")
async def async_io_task(text: str, delay_ms: float = 10) -> str:
    """Async I/O-bound task."""
    await asyncio.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


@node(output_name="result")
def sync_io_task(text: str, delay_ms: float = 10) -> str:
    """Sync I/O-bound task."""
    time.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


# ==================== Test Data ====================

def generate_test_texts(count: int) -> List[str]:
    """Generate test text data."""
    return [
        f"The quick brown fox jumps over the lazy dog number {i}. "
        f"This is sentence {i} with some more words to process."
        for i in range(count)
    ]


# ==================== Tests ====================

def test_async_daft_engine(texts: List[str], delay_ms: float = 10):
    """Test async function support in DaftEngine."""
    print(f"\n{'='*60}")
    print(f"TEST: Async Function Support in DaftEngine")
    print(f"{'='*60}")
    
    pipeline = Pipeline(
        nodes=[async_io_task],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    print(f"Processing {len(texts)} items with async functions...")
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "delay_ms": delay_ms},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    sequential_time = len(texts) * (delay_ms / 1000)
    speedup = sequential_time / elapsed
    
    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Sequential would take: {sequential_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Expected: ~37x (based on benchmark)")
    
    return elapsed, speedup


def test_sync_batch_daft_engine(texts: List[str], delay_ms: float = 10):
    """Test ThreadPoolExecutor in batch UDF."""
    print(f"\n{'='*60}")
    print(f"TEST: Sync Batch UDF with ThreadPoolExecutor")
    print(f"{'='*60}")
    
    pipeline = Pipeline(
        nodes=[sync_io_task],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    print(f"Processing {len(texts)} items with sync functions...")
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "delay_ms": delay_ms},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    sequential_time = len(texts) * (delay_ms / 1000)
    speedup = sequential_time / elapsed
    
    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Sequential would take: {sequential_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Expected: ~10x (based on benchmark)")
    
    return elapsed, speedup


def test_dask_baseline(texts: List[str], delay_ms: float = 10):
    """Test DaskEngine as baseline."""
    print(f"\n{'='*60}")
    print(f"BASELINE: DaskEngine (threads)")
    print(f"{'='*60}")
    
    pipeline = Pipeline(
        nodes=[sync_io_task],
        engine=DaskEngine(scheduler="threads", workload_type="io")
    )
    
    print(f"Processing {len(texts)} items with DaskEngine...")
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "delay_ms": delay_ms},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    sequential_time = len(texts) * (delay_ms / 1000)
    speedup = sequential_time / elapsed
    
    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Sequential would take: {sequential_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    return elapsed, speedup


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("IMPROVED DAFT ENGINE VALIDATION")
    print("="*60)
    
    # Test with 100 items (same as benchmark)
    texts = generate_test_texts(100)
    delay_ms = 10
    
    # Run tests
    dask_time, dask_speedup = test_dask_baseline(texts, delay_ms)
    async_time, async_speedup = test_async_daft_engine(texts, delay_ms)
    sync_time, sync_speedup = test_sync_batch_daft_engine(texts, delay_ms)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<40} {'Time (s)':<12} {'Speedup'}")
    print(f"{'-'*60}")
    print(f"{'DaskEngine (baseline)':<40} {dask_time:.3f}        {dask_speedup:.2f}x")
    print(f"{'DaftEngine + async functions':<40} {async_time:.3f}        {async_speedup:.2f}x")
    print(f"{'DaftEngine + sync batch (ThreadPool)':<40} {sync_time:.3f}        {sync_speedup:.2f}x")
    print(f"{'='*60}")
    
    # Validation
    print("\nVALIDATION:")
    if async_speedup > 20:
        print(f"✓ Async speedup is EXCELLENT ({async_speedup:.1f}x > 20x)")
    elif async_speedup > 5:
        print(f"✓ Async speedup is GOOD ({async_speedup:.1f}x > 5x)")
    else:
        print(f"✗ Async speedup is LOW ({async_speedup:.1f}x)")
    
    if sync_speedup > 5:
        print(f"✓ Sync batch speedup is GOOD ({sync_speedup:.1f}x > 5x)")
    elif sync_speedup > 2:
        print(f"~ Sync batch speedup is OK ({sync_speedup:.1f}x > 2x)")
    else:
        print(f"✗ Sync batch speedup is LOW ({sync_speedup:.1f}x)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


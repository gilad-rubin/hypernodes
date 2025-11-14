#!/usr/bin/env python3
"""
Debug async DaftEngine performance

Compare pure Daft async vs DaftEngine async to find the bottleneck.
"""

import asyncio
import time
import daft
from typing import List

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# ==================== Test Data ====================

def generate_test_texts(count: int) -> List[str]:
    return [f"Text item {i}" for i in range(count)]


# ==================== Pure Daft Test ====================

def test_pure_daft_async(texts: List[str], delay_ms: float = 10):
    """Pure Daft async (baseline - should be ~37x)."""
    print(f"\n{'='*60}")
    print("PURE DAFT ASYNC (Baseline)")
    print(f"{'='*60}")
    
    @daft.func
    async def async_process(text: str) -> str:
        await asyncio.sleep(delay_ms / 1000)
        return f"processed: {text[:20]}..."
    
    start = time.perf_counter()
    df = daft.from_pydict({"text": texts})
    df = df.with_column("result", async_process(daft.col("text")))
    result_df = df.collect()
    elapsed = time.perf_counter() - start
    
    sequential_time = len(texts) * (delay_ms / 1000)
    speedup = sequential_time / elapsed
    
    print(f"✓ Time: {elapsed:.3f}s | Speedup: {speedup:.2f}x")
    return elapsed, speedup


# ==================== DaftEngine Test ====================

@node(output_name="result")
async def async_io_task(text: str, delay_ms: float = 10) -> str:
    """Async I/O-bound task."""
    await asyncio.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


def test_daft_engine_async(texts: List[str], delay_ms: float = 10, use_batch: bool = True):
    """DaftEngine async."""
    print(f"\n{'='*60}")
    print(f"DAFT ENGINE ASYNC (use_batch_udf={use_batch})")
    print(f"{'='*60}")
    
    pipeline = Pipeline(
        nodes=[async_io_task],
        engine=DaftEngine(use_batch_udf=use_batch)
    )
    
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "delay_ms": delay_ms},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    sequential_time = len(texts) * (delay_ms / 1000)
    speedup = sequential_time / elapsed
    
    print(f"✓ Time: {elapsed:.3f}s | Speedup: {speedup:.2f}x")
    return elapsed, speedup


# ==================== Main ====================

def main():
    """Run debugging tests."""
    print("\n" + "="*60)
    print("ASYNC DAFT DEBUGGING")
    print("="*60)
    
    texts = generate_test_texts(100)
    delay_ms = 10
    
    # Test pure Daft (expected: ~37x)
    pure_time, pure_speedup = test_pure_daft_async(texts, delay_ms)
    
    # Test DaftEngine with batch UDF enabled
    engine_batch_time, engine_batch_speedup = test_daft_engine_async(texts, delay_ms, use_batch=True)
    
    # Test DaftEngine with batch UDF disabled
    engine_no_batch_time, engine_no_batch_speedup = test_daft_engine_async(texts, delay_ms, use_batch=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Strategy':<45} {'Time':<10} {'Speedup'}")
    print(f"{'-'*60}")
    print(f"{'Pure Daft async @daft.func':<45} {pure_time:.3f}s     {pure_speedup:.2f}x")
    print(f"{'DaftEngine (use_batch_udf=True)':<45} {engine_batch_time:.3f}s     {engine_batch_speedup:.2f}x")
    print(f"{'DaftEngine (use_batch_udf=False)':<45} {engine_no_batch_time:.3f}s     {engine_no_batch_speedup:.2f}x")
    print(f"{'='*60}")
    
    # Analysis
    print("\nANALYSIS:")
    overhead = engine_batch_time - pure_time
    print(f"DaftEngine overhead: {overhead:.3f}s ({overhead/pure_time*100:.1f}%)")
    
    if engine_batch_speedup < pure_speedup * 0.5:
        print(f"⚠️  DaftEngine async is significantly slower than pure Daft")
        print(f"   Possible causes:")
        print(f"   - MapPlanner overhead")
        print(f"   - DataFrame construction overhead")
        print(f"   - Collect/extract overhead")
    else:
        print(f"✓ DaftEngine async performance is acceptable")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Comprehensive Daft Parallelism Benchmark

Tests different execution strategies for both I/O-bound and CPU-bound workloads:
1. DaskEngine (threads) - baseline for I/O
2. DaskEngine (processes) - baseline for CPU
3. Pure Daft sync @daft.func (row-wise)
4. Pure Daft async @daft.func (row-wise)
5. Pure Daft @daft.func.batch (hand-written)
6. Current DaftEngine (for comparison)

Goal: Determine which strategy provides "for free" parallelism comparable to Dask's 7-8x speedup.
"""

import asyncio
import time
from typing import List, Dict, Any
import daft
from daft import DataType, Series

from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine, DaftEngine


# ==================== Test Data ====================

def generate_test_texts(count: int) -> List[str]:
    """Generate test text data."""
    return [
        f"The quick brown fox jumps over the lazy dog number {i}. "
        f"This is sentence {i} with some more words to process. "
        f"Additional text for item {i} to make processing more realistic."
        for i in range(count)
    ]


# ==================== I/O-Bound Workloads ====================

@node(output_name="result")
def sync_io_task(text: str, delay_ms: float = 10) -> str:
    """Simulate I/O-bound operation (sync version with sleep)."""
    time.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


@node(output_name="result")
async def async_io_task(text: str, delay_ms: float = 10) -> str:
    """Simulate I/O-bound operation (async version)."""
    await asyncio.sleep(delay_ms / 1000)
    return f"processed: {text[:20]}..."


# ==================== CPU-Bound Workloads ====================

@node(output_name="chunks")
def cpu_text_chunking(text: str, chunk_size: int = 10) -> List[str]:
    """CPU-bound: Split text into chunks."""
    words = text.lower().split()
    chunks = [
        " ".join(words[i:i + chunk_size]) 
        for i in range(0, len(words), chunk_size)
    ]
    return chunks


@node(output_name="word_count")
def cpu_word_count(text: str) -> int:
    """CPU-bound: Count words in text."""
    return len(text.lower().split())


# ==================== Pure Daft Implementations ====================

def test_pure_daft_sync_rowwise(texts: List[str], delay_ms: float = 10) -> tuple:
    """Test 1: Pure Daft with sync @daft.func (row-wise)."""
    
    @daft.func
    def daft_sync_io(text: str) -> str:
        time.sleep(delay_ms / 1000)
        return f"processed: {text[:20]}..."
    
    start = time.perf_counter()
    df = daft.from_pydict({"text": texts})
    df = df.with_column("result", daft_sync_io(daft.col("text")))
    result_df = df.collect()
    elapsed = time.perf_counter() - start
    
    return elapsed, len(result_df)


def test_pure_daft_async_rowwise(texts: List[str], delay_ms: float = 10) -> tuple:
    """Test 2: Pure Daft with async @daft.func (row-wise)."""
    
    @daft.func
    async def daft_async_io(text: str) -> str:
        await asyncio.sleep(delay_ms / 1000)
        return f"processed: {text[:20]}..."
    
    start = time.perf_counter()
    df = daft.from_pydict({"text": texts})
    df = df.with_column("result", daft_async_io(daft.col("text")))
    result_df = df.collect()
    elapsed = time.perf_counter() - start
    
    return elapsed, len(result_df)


def test_pure_daft_batch_native(texts: List[str], delay_ms: float = 10) -> tuple:
    """Test 3: Pure Daft with @daft.func.batch (native batch processing)."""
    
    @daft.func.batch(return_dtype=DataType.string())
    def daft_batch_io(text_series: Series) -> Series:
        """Process entire batch at once (not row-by-row)."""
        import concurrent.futures
        
        def process_one(text: str) -> str:
            time.sleep(delay_ms / 1000)
            return f"processed: {text[:20]}..."
        
        texts_list = text_series.to_pylist()
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_one, texts_list))
        
        return Series.from_pylist(results)
    
    start = time.perf_counter()
    df = daft.from_pydict({"text": texts})
    df = df.with_column("result", daft_batch_io(daft.col("text")))
    result_df = df.collect()
    elapsed = time.perf_counter() - start
    
    return elapsed, len(result_df)


def test_pure_daft_cpu_chunking(texts: List[str]) -> tuple:
    """Test 4: Pure Daft with CPU-bound text chunking."""
    
    @daft.func
    def daft_chunk_text(text: str) -> List[str]:
        words = text.lower().split()
        chunks = [
            " ".join(words[i:i + 10]) 
            for i in range(0, len(words), 10)
        ]
        return chunks
    
    start = time.perf_counter()
    df = daft.from_pydict({"text": texts})
    df = df.with_column("chunks", daft_chunk_text(daft.col("text")))
    result_df = df.collect()
    elapsed = time.perf_counter() - start
    
    return elapsed, len(result_df)


# ==================== HyperNodes DaskEngine Tests ====================

def test_dask_engine_threads_io(texts: List[str], delay_ms: float = 10) -> tuple:
    """Test 5: DaskEngine with threads (I/O-bound baseline)."""
    
    pipeline = Pipeline(
        nodes=[sync_io_task],
        engine=DaskEngine(scheduler="threads", workload_type="io")
    )
    
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "delay_ms": delay_ms},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    return elapsed, len(results)


def test_dask_engine_processes_cpu(texts: List[str]) -> tuple:
    """Test 6: DaskEngine with processes (CPU-bound baseline)."""
    
    pipeline = Pipeline(
        nodes=[cpu_text_chunking],
        engine=DaskEngine(scheduler="processes", workload_type="cpu")
    )
    
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "chunk_size": 10},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    return elapsed, len(results)


def test_dask_engine_threads_cpu(texts: List[str]) -> tuple:
    """Test 7: DaskEngine with threads (CPU-bound for comparison)."""
    
    pipeline = Pipeline(
        nodes=[cpu_word_count],
        engine=DaskEngine(scheduler="threads", workload_type="cpu")
    )
    
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    return elapsed, len(results)


# ==================== HyperNodes DaftEngine Tests ====================

def test_current_daft_engine_io(texts: List[str], delay_ms: float = 10) -> tuple:
    """Test 8: Current DaftEngine implementation (I/O-bound)."""
    
    pipeline = Pipeline(
        nodes=[sync_io_task],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts, "delay_ms": delay_ms},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    return elapsed, len(results)


def test_current_daft_engine_cpu(texts: List[str]) -> tuple:
    """Test 9: Current DaftEngine implementation (CPU-bound)."""
    
    pipeline = Pipeline(
        nodes=[cpu_word_count],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    start = time.perf_counter()
    results = pipeline.map(
        inputs={"text": texts},
        map_over="text"
    )
    elapsed = time.perf_counter() - start
    
    return elapsed, len(results)


# ==================== Sequential Baseline ====================

def test_sequential_io(texts: List[str], delay_ms: float = 10) -> tuple:
    """Sequential baseline for I/O-bound."""
    start = time.perf_counter()
    results = []
    for text in texts:
        time.sleep(delay_ms / 1000)
        results.append(f"processed: {text[:20]}...")
    elapsed = time.perf_counter() - start
    return elapsed, len(results)


def test_sequential_cpu(texts: List[str]) -> tuple:
    """Sequential baseline for CPU-bound."""
    start = time.perf_counter()
    results = []
    for text in texts:
        results.append(len(text.lower().split()))
    elapsed = time.perf_counter() - start
    return elapsed, len(results)


# ==================== Benchmark Runner ====================

def run_benchmark(test_func, name: str, texts: List[str], **kwargs) -> Dict[str, Any]:
    """Run a single benchmark test."""
    print(f"\n  Running: {name}...", end=" ", flush=True)
    
    try:
        elapsed, count = test_func(texts, **kwargs)
        print(f"✓ ({elapsed:.3f}s)")
        return {
            "name": name,
            "elapsed": elapsed,
            "count": count,
            "success": True
        }
    except Exception as e:
        print(f"✗ ({str(e)[:50]})")
        return {
            "name": name,
            "elapsed": None,
            "count": None,
            "success": False,
            "error": str(e)
        }


def print_results_table(results: List[Dict[str, Any]], baseline_time: float, category: str):
    """Print results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"{category} RESULTS")
    print(f"{'='*80}")
    print(f"{'Strategy':<45} {'Time (s)':<12} {'Speedup':<10} {'Status'}")
    print(f"{'-'*80}")
    
    for r in results:
        if r["success"]:
            speedup = baseline_time / r["elapsed"] if r["elapsed"] > 0 else 0
            speedup_str = f"{speedup:.2f}x"
            status = "✓"
        else:
            speedup_str = "N/A"
            status = "✗"
        
        time_str = f"{r['elapsed']:.3f}" if r["elapsed"] else "FAILED"
        print(f"{r['name']:<45} {time_str:<12} {speedup_str:<10} {status}")


def main():
    """Run all benchmarks."""
    print("\n" + "="*80)
    print("DAFT PARALLELISM COMPREHENSIVE BENCHMARK")
    print("="*80)
    
    # Test parameters
    scales = [10, 50, 100]  # Different dataset sizes
    io_delay_ms = 10  # 10ms simulated I/O delay
    
    for scale in scales:
        print(f"\n{'#'*80}")
        print(f"# SCALE: {scale} items")
        print(f"{'#'*80}")
        
        texts = generate_test_texts(scale)
        
        # ==================== I/O-BOUND TESTS ====================
        print(f"\n{'-'*80}")
        print(f"I/O-BOUND WORKLOAD (simulated {io_delay_ms}ms delay per item)")
        print(f"{'-'*80}")
        
        io_results = []
        
        # Baseline
        baseline_io = run_benchmark(
            test_sequential_io, 
            "Sequential (baseline)",
            texts, 
            delay_ms=io_delay_ms
        )
        io_results.append(baseline_io)
        baseline_io_time = baseline_io["elapsed"] if baseline_io["success"] else 1.0
        
        # DaskEngine tests
        io_results.append(run_benchmark(
            test_dask_engine_threads_io,
            "DaskEngine (threads)",
            texts,
            delay_ms=io_delay_ms
        ))
        
        # Pure Daft tests
        io_results.append(run_benchmark(
            test_pure_daft_sync_rowwise,
            "Pure Daft: sync @daft.func (row-wise)",
            texts,
            delay_ms=io_delay_ms
        ))
        
        io_results.append(run_benchmark(
            test_pure_daft_async_rowwise,
            "Pure Daft: async @daft.func (row-wise)",
            texts,
            delay_ms=io_delay_ms
        ))
        
        io_results.append(run_benchmark(
            test_pure_daft_batch_native,
            "Pure Daft: @daft.func.batch (ThreadPool)",
            texts,
            delay_ms=io_delay_ms
        ))
        
        # Current DaftEngine
        io_results.append(run_benchmark(
            test_current_daft_engine_io,
            "Current DaftEngine (batch UDF)",
            texts,
            delay_ms=io_delay_ms
        ))
        
        print_results_table(io_results, baseline_io_time, "I/O-BOUND")
        
        # ==================== CPU-BOUND TESTS ====================
        print(f"\n{'-'*80}")
        print(f"CPU-BOUND WORKLOAD (text processing)")
        print(f"{'-'*80}")
        
        cpu_results = []
        
        # Baseline
        baseline_cpu = run_benchmark(
            test_sequential_cpu,
            "Sequential (baseline)",
            texts
        )
        cpu_results.append(baseline_cpu)
        baseline_cpu_time = baseline_cpu["elapsed"] if baseline_cpu["success"] else 1.0
        
        # DaskEngine tests
        cpu_results.append(run_benchmark(
            test_dask_engine_threads_cpu,
            "DaskEngine (threads)",
            texts
        ))
        
        cpu_results.append(run_benchmark(
            test_dask_engine_processes_cpu,
            "DaskEngine (processes)",
            texts
        ))
        
        # Pure Daft tests
        cpu_results.append(run_benchmark(
            test_pure_daft_cpu_chunking,
            "Pure Daft: sync @daft.func (text chunking)",
            texts
        ))
        
        # Current DaftEngine
        cpu_results.append(run_benchmark(
            test_current_daft_engine_cpu,
            "Current DaftEngine (batch UDF)",
            texts
        ))
        
        print_results_table(cpu_results, baseline_cpu_time, "CPU-BOUND")
    
    # ==================== SUMMARY ====================
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print("\nKey Findings:")
    print("1. Check if async @daft.func provides concurrency for I/O tasks")
    print("2. Compare Pure Daft row-wise vs DaskEngine parallelism")
    print("3. Evaluate if @daft.func.batch with ThreadPool beats sequential loop")
    print("4. Identify which strategy gives 'for free' parallelism")
    print("\nNext Steps:")
    print("- Analyze which Daft strategies match DaskEngine's 7-8x speedup")
    print("- Fix DaftEngine batch wrapper based on findings")
    print("- Add async support to DaftEngine if async @daft.func performs well")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


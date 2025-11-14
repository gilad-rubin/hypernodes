#!/usr/bin/env python3
"""
Benchmark CrossEncoder Reranking Performance with Daft
========================================================

This script tests multiple optimization strategies for the reranking step:
1. Baseline (row-wise processing)
2. Batch UDF processing
3. Async row-wise processing
4. Parallelization with max_concurrency
5. Combined optimizations

Uses native Daft (@daft.cls, @daft.method.batch) without HyperNodes.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import daft
import pandas as pd
import psutil
from daft import DataType, Series
from sentence_transformers import CrossEncoder

# ==================== Data Loading ====================


def load_benchmark_data(n_queries: int = 10, n_candidates_per_query: int = 30):
    """Load queries and candidates for reranking benchmark."""

    # Load queries
    queries_df = pd.read_parquet("data/sample_10/dev.parquet")
    query_df = (
        queries_df[["query_uuid", "query_text"]].drop_duplicates().head(n_queries)
    )

    # Load corpus
    corpus_df = pd.read_parquet("data/sample_10/corpus.parquet")

    # Create query-candidate pairs
    # For each query, sample N random candidate passages
    nested_data = []
    flat_data = []

    for _, query_row in query_df.iterrows():
        query_uuid = query_row["query_uuid"]
        query_text = query_row["query_text"]

        # Sample random candidates
        candidates = corpus_df.sample(n=min(n_candidates_per_query, len(corpus_df)))
        candidate_texts = candidates["passage"].tolist()
        candidate_uuids = candidates["uuid"].tolist()

        # Nested format for row-wise processing
        nested_data.append(
            {
                "query_uuid": query_uuid,
                "query_text": query_text,
                "candidate_texts": candidate_texts,
                "candidate_uuids": candidate_uuids,
                "n_candidates": len(candidate_texts),
            }
        )

        # Flat format for batch processing
        for cand_text, cand_uuid in zip(candidate_texts, candidate_uuids):
            flat_data.append(
                {
                    "query_uuid": query_uuid,
                    "query_text": query_text,
                    "candidate_text": cand_text,
                    "candidate_uuid": cand_uuid,
                }
            )

    print(
        f"Loaded {len(nested_data)} queries with ~{n_candidates_per_query} candidates each"
    )
    print(f"Total query-candidate pairs: {len(flat_data)}")

    return nested_data, flat_data


# ==================== Reranker Implementations ====================


# 1. Baseline: Row-wise Processing
@daft.cls
class RowwiseReranker:
    """Baseline reranker - processes one query at a time."""

    def __init__(self, model_name: str):
        print(f"[RowwiseReranker] Loading {model_name}...")
        self.model = CrossEncoder(model_name)

    def __call__(self, query_text: str, candidate_texts: list) -> list:
        """Rerank candidates for a single query."""
        pairs = [[query_text, cand] for cand in candidate_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist()


# 2. Batch UDF Processing - works with FLAT data (one row per query-candidate pair)
@daft.cls
class BatchReranker:
    """Batch reranker - processes multiple query-candidate pairs simultaneously."""

    def __init__(self, model_name: str, cross_encoder_batch_size: int = 32):
        print(
            f"[BatchReranker] Loading {model_name} (batch_size={cross_encoder_batch_size})..."
        )
        self.model = CrossEncoder(model_name)
        self.cross_encoder_batch_size = cross_encoder_batch_size

    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        """Score query-candidate pairs in batch."""
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()

        # Create pairs
        pairs = [[q, c] for q, c in zip(queries, candidates)]

        # Score all pairs in one batch
        scores = self.model.predict(
            pairs, batch_size=self.cross_encoder_batch_size, show_progress_bar=False
        )

        return scores.tolist()


# 3. Async Row-wise Processing
@daft.cls
class AsyncReranker:
    """Async reranker - concurrent processing of queries."""

    def __init__(self, model_name: str):
        print(f"[AsyncReranker] Loading {model_name}...")
        self.model = CrossEncoder(model_name)

    async def __call__(self, query_text: str, candidate_texts: list) -> list:
        """Async rerank - runs concurrently across rows."""
        # Note: CrossEncoder.predict is CPU-bound, so async won't help much
        # but we test it anyway to demonstrate the pattern
        pairs = [[query_text, cand] for cand in candidate_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist()


# 4. Concurrent Reranker (with max_concurrency)
@daft.cls(max_concurrency=4)
class ConcurrentReranker:
    """Concurrent reranker - multiple instances processing in parallel."""

    def __init__(self, model_name: str):
        print(f"[ConcurrentReranker] Loading {model_name}...")
        self.model = CrossEncoder(model_name)

    def __call__(self, query_text: str, candidate_texts: list) -> list:
        """Rerank with concurrency control."""
        pairs = [[query_text, cand] for cand in candidate_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist()


# 5. Optimized: Batch + Concurrency - works with FLAT data
@daft.cls(max_concurrency=2)
class OptimizedReranker:
    """Optimized reranker - combines batch processing with concurrency."""

    def __init__(self, model_name: str, cross_encoder_batch_size: int = 64):
        print(
            f"[OptimizedReranker] Loading {model_name} (batch_size={cross_encoder_batch_size})..."
        )
        self.model = CrossEncoder(model_name)
        self.cross_encoder_batch_size = cross_encoder_batch_size

    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        """Optimized batch scoring."""
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()

        pairs = [[q, c] for q, c in zip(queries, candidates)]

        scores = self.model.predict(
            pairs, batch_size=self.cross_encoder_batch_size, show_progress_bar=False
        )

        return scores.tolist()


# ==================== Resource Profiling ====================


class ResourceMonitor:
    """Monitor CPU and memory usage during execution."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.cpu_per_core_samples = []
        self.thread = None
        self.process = psutil.Process()

    def _monitor(self):
        """Background monitoring thread."""
        while self.monitoring:
            # Overall CPU usage (percentage)
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)

            # Per-core CPU usage (system-wide)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.cpu_per_core_samples.append(cpu_per_core)

            # Memory usage (RSS in MB)
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.cpu_per_core_samples = []

        # Initialize CPU percent (first call returns 0)
        self.process.cpu_percent(interval=None)

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> Dict:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if not self.cpu_samples:
            return {
                "cpu_mean": 0,
                "cpu_max": 0,
                "cpu_cores_utilized": 0,
                "memory_mean_mb": 0,
                "memory_max_mb": 0,
            }

        # Calculate per-core utilization (average across all samples)
        num_cores = psutil.cpu_count()
        core_utilization = [0] * num_cores

        for sample in self.cpu_per_core_samples:
            for i, util in enumerate(sample):
                core_utilization[i] += util

        if self.cpu_per_core_samples:
            core_utilization = [
                u / len(self.cpu_per_core_samples) for u in core_utilization
            ]

        # Count how many cores were significantly utilized (>10%)
        cores_utilized = sum(1 for u in core_utilization if u > 10)

        return {
            "cpu_mean": sum(self.cpu_samples) / len(self.cpu_samples),
            "cpu_max": max(self.cpu_samples),
            "cpu_cores_utilized": cores_utilized,
            "cpu_per_core": core_utilization,
            "memory_mean_mb": sum(self.memory_samples) / len(self.memory_samples),
            "memory_max_mb": max(self.memory_samples),
            "num_cores": num_cores,
        }


# ==================== Benchmark Functions ====================


def benchmark_rowwise_strategy(
    strategy_name: str, df: daft.DataFrame, reranker_instance, total_pairs: int
):
    """Benchmark a row-wise reranking strategy (nested data)."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {strategy_name}")
    print(f"{'=' * 60}")

    # Start resource monitoring
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    start = time.perf_counter()

    # Apply reranking (row-wise)
    result_df = df.with_column(
        "scores", reranker_instance(df["query_text"], df["candidate_texts"])
    )

    # Collect to force execution
    result = result_df.collect()

    elapsed = time.perf_counter() - start

    # Stop monitoring and get stats
    resource_stats = monitor.stop()

    pairs_per_sec = total_pairs / elapsed if elapsed > 0 else 0

    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Throughput: {pairs_per_sec:.1f} pairs/sec")
    print(
        f"  CPU: {resource_stats['cpu_mean']:.1f}% avg, {resource_stats['cpu_max']:.1f}% peak"
    )
    print(
        f"  Cores utilized: {resource_stats['cpu_cores_utilized']}/{resource_stats['num_cores']}"
    )
    print(
        f"  Memory: {resource_stats['memory_mean_mb']:.1f} MB avg, {resource_stats['memory_max_mb']:.1f} MB peak"
    )

    return {
        "strategy": strategy_name,
        "time": elapsed,
        "pairs_per_sec": pairs_per_sec,
        "total_pairs": total_pairs,
        "cpu_mean": resource_stats["cpu_mean"],
        "cpu_max": resource_stats["cpu_max"],
        "cores_utilized": resource_stats["cpu_cores_utilized"],
        "num_cores": resource_stats["num_cores"],
        "memory_mean_mb": resource_stats["memory_mean_mb"],
        "memory_max_mb": resource_stats["memory_max_mb"],
    }


def benchmark_batch_strategy(
    strategy_name: str,
    df: daft.DataFrame,
    reranker_instance,
    method_name: str,
    total_pairs: int,
):
    """Benchmark a batch reranking strategy (flat data)."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {strategy_name}")
    print(f"{'=' * 60}")

    # Start resource monitoring
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    start = time.perf_counter()

    # Apply batch reranking (flat data - one row per pair)
    result_df = df.with_column(
        "score",
        getattr(reranker_instance, method_name)(df["query_text"], df["candidate_text"]),
    )

    # Collect to force execution
    result = result_df.collect()

    elapsed = time.perf_counter() - start

    # Stop monitoring and get stats
    resource_stats = monitor.stop()

    pairs_per_sec = total_pairs / elapsed if elapsed > 0 else 0

    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Throughput: {pairs_per_sec:.1f} pairs/sec")
    print(
        f"  CPU: {resource_stats['cpu_mean']:.1f}% avg, {resource_stats['cpu_max']:.1f}% peak"
    )
    print(
        f"  Cores utilized: {resource_stats['cpu_cores_utilized']}/{resource_stats['num_cores']}"
    )
    print(
        f"  Memory: {resource_stats['memory_mean_mb']:.1f} MB avg, {resource_stats['memory_max_mb']:.1f} MB peak"
    )

    return {
        "strategy": strategy_name,
        "time": elapsed,
        "pairs_per_sec": pairs_per_sec,
        "total_pairs": total_pairs,
        "cpu_mean": resource_stats["cpu_mean"],
        "cpu_max": resource_stats["cpu_max"],
        "cores_utilized": resource_stats["cpu_cores_utilized"],
        "num_cores": resource_stats["num_cores"],
        "memory_mean_mb": resource_stats["memory_mean_mb"],
        "memory_max_mb": resource_stats["memory_max_mb"],
    }


def print_results_table(results: List[dict]):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Find baseline time
    baseline_time = next(
        (r["time"] for r in results if "Baseline" in r["strategy"]), results[0]["time"]
    )

    # Performance table
    print(f"\n{'Strategy':<35} {'Time (s)':>10} {'Speedup':>10} {'Pairs/sec':>12}")
    print("-" * 100)

    for result in results:
        speedup = baseline_time / result["time"] if result["time"] > 0 else 0
        print(
            f"{result['strategy']:<35} {result['time']:>10.3f} {speedup:>9.1f}x {result['pairs_per_sec']:>12.1f}"
        )

    # Resource utilization table
    print("\n" + "=" * 100)
    print("RESOURCE UTILIZATION")
    print("=" * 100)
    print(
        f"\n{'Strategy':<35} {'CPU Avg':>10} {'CPU Peak':>10} {'Cores':>8} {'Mem Avg':>12} {'Mem Peak':>12}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result['strategy']:<35} "
            f"{result['cpu_mean']:>9.1f}% "
            f"{result['cpu_max']:>9.1f}% "
            f"{result['cores_utilized']:>3}/{result['num_cores']:<3} "
            f"{result['memory_mean_mb']:>10.1f}MB "
            f"{result['memory_max_mb']:>10.1f}MB"
        )

    print("=" * 100)

    # Find best strategy
    best = max(results, key=lambda x: x["pairs_per_sec"])
    print(f"\n🏆 WINNER: {best['strategy']}")
    print(
        f"   {best['pairs_per_sec']:.1f} pairs/sec ({baseline_time / best['time']:.1f}x faster than baseline)"
    )
    print(
        f"   CPU: {best['cpu_mean']:.1f}% avg, using {best['cores_utilized']}/{best['num_cores']} cores"
    )
    print(f"   Memory: {best['memory_mean_mb']:.1f} MB avg")

    # Analysis
    print("\n" + "=" * 100)
    print("RESOURCE UTILIZATION ANALYSIS")
    print("=" * 100)

    num_cores = results[0]["num_cores"]
    max_cores_used = max(r["cores_utilized"] for r in results)
    avg_cpu_utilization = sum(r["cpu_mean"] for r in results) / len(results)

    print(f"System: {num_cores} CPU cores available")
    print(
        f"Maximum cores utilized: {max_cores_used}/{num_cores} ({max_cores_used / num_cores * 100:.1f}%)"
    )
    print(f"Average CPU utilization across strategies: {avg_cpu_utilization:.1f}%")

    if max_cores_used < num_cores * 0.5:
        print("\n⚠️  LOW PARALLELISM: Using less than 50% of available cores")
        print("   → This workload is not effectively utilizing multi-core parallelism")
        print("   → CrossEncoder inference is likely the bottleneck")
    elif max_cores_used < num_cores * 0.8:
        print("\n⚡ MODERATE PARALLELISM: Using 50-80% of available cores")
        print("   → Some parallelism achieved, but room for improvement")
    else:
        print("\n✓ GOOD PARALLELISM: Using >80% of available cores")

    # Memory analysis
    max_memory = max(r["memory_max_mb"] for r in results)
    min_memory = min(r["memory_max_mb"] for r in results)

    print(f"\nMemory usage range: {min_memory:.1f} MB - {max_memory:.1f} MB")
    if max_memory - min_memory > 500:
        print("⚠️  Significant memory variance between strategies")
    else:
        print("✓ Consistent memory usage across strategies")

    print("=" * 100)


# ==================== Main ====================


def main():
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_QUERIES = 10
    N_CANDIDATES = 30

    print("=" * 80)
    print("CROSSENCODER RERANKING OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Queries: {N_QUERIES}")
    print(f"Candidates per query: {N_CANDIDATES}")
    print("=" * 80)

    # Load data
    print("\nLoading benchmark data...")
    nested_data, flat_data = load_benchmark_data(
        n_queries=N_QUERIES, n_candidates_per_query=N_CANDIDATES
    )
    total_pairs = len(flat_data)

    # Create DataFrames - nested for row-wise, flat for batch
    df_nested = daft.from_pydict(
        {
            "query_uuid": [d["query_uuid"] for d in nested_data],
            "query_text": [d["query_text"] for d in nested_data],
            "candidate_texts": [d["candidate_texts"] for d in nested_data],
            "candidate_uuids": [d["candidate_uuids"] for d in nested_data],
            "n_candidates": [d["n_candidates"] for d in nested_data],
        }
    )

    df_flat = daft.from_pydict(
        {
            "query_uuid": [d["query_uuid"] for d in flat_data],
            "query_text": [d["query_text"] for d in flat_data],
            "candidate_text": [d["candidate_text"] for d in flat_data],
            "candidate_uuid": [d["candidate_uuid"] for d in flat_data],
        }
    )

    results = []

    # 1. Baseline: Row-wise
    print("\n" + ">" * 80)
    print("STRATEGY 1: Baseline (Row-wise Processing)")
    print(">" * 80)
    rowwise = RowwiseReranker(MODEL_NAME)
    results.append(
        benchmark_rowwise_strategy(
            "1. Baseline (Row-wise)", df_nested, rowwise, total_pairs
        )
    )

    # 2. Batch UDF (batch_size=32)
    print("\n" + ">" * 80)
    print("STRATEGY 2: Batch UDF (batch_size=32)")
    print(">" * 80)
    batch32 = BatchReranker(MODEL_NAME, cross_encoder_batch_size=32)
    results.append(
        benchmark_batch_strategy(
            "2. Batch UDF (batch_size=32)", df_flat, batch32, "score_pairs", total_pairs
        )
    )

    # 3. Batch UDF (batch_size=64)
    print("\n" + ">" * 80)
    print("STRATEGY 3: Batch UDF (batch_size=64)")
    print(">" * 80)
    batch64 = BatchReranker(MODEL_NAME, cross_encoder_batch_size=64)
    results.append(
        benchmark_batch_strategy(
            "3. Batch UDF (batch_size=64)", df_flat, batch64, "score_pairs", total_pairs
        )
    )

    # 4. Batch UDF (batch_size=128)
    print("\n" + ">" * 80)
    print("STRATEGY 4: Batch UDF (batch_size=128)")
    print(">" * 80)
    batch128 = BatchReranker(MODEL_NAME, cross_encoder_batch_size=128)
    results.append(
        benchmark_batch_strategy(
            "4. Batch UDF (batch_size=128)",
            df_flat,
            batch128,
            "score_pairs",
            total_pairs,
        )
    )

    # 5. Async (for comparison)
    print("\n" + ">" * 80)
    print("STRATEGY 5: Async Row-wise")
    print(">" * 80)
    async_reranker = AsyncReranker(MODEL_NAME)
    results.append(
        benchmark_rowwise_strategy(
            "5. Async Row-wise", df_nested, async_reranker, total_pairs
        )
    )

    # 6. Concurrent (max_concurrency=4)
    print("\n" + ">" * 80)
    print("STRATEGY 6: Concurrent (max_concurrency=4)")
    print(">" * 80)
    concurrent = ConcurrentReranker(MODEL_NAME)
    results.append(
        benchmark_rowwise_strategy(
            "6. Concurrent (max_concurrency=4)", df_nested, concurrent, total_pairs
        )
    )

    # 7. Optimized: Batch + Concurrency
    print("\n" + ">" * 80)
    print("STRATEGY 7: Optimized (Batch + Concurrency)")
    print(">" * 80)
    optimized = OptimizedReranker(MODEL_NAME, cross_encoder_batch_size=64)
    results.append(
        benchmark_batch_strategy(
            "7. Optimized (batch=64, concurrency=2)",
            df_flat,
            optimized,
            "score_pairs",
            total_pairs,
        )
    )

    # Print final results table
    print_results_table(results)

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. Use @daft.method.batch for processing multiple queries together")
    print("2. Set CrossEncoder batch_size to 64-128 for best throughput")
    print("3. Async provides minimal benefit (CPU-bound workload)")
    print("4. Consider max_concurrency only if processing very large batches")
    print("5. For production: Use BatchReranker with batch_size=64")
    print("=" * 80)


if __name__ == "__main__":
    main()

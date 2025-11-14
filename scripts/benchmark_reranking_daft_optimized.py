#!/usr/bin/env python3
"""
CrossEncoder Reranking with Daft Best Practices
================================================

Applies optimization patterns from Daft's AI functions documentation:
1. Model compilation with torch.compile()
2. Proper batch sizing
3. GPU detection and usage
4. Efficient tensor handling
5. Process isolation for stability

Based on: https://www.getdaft.io/modalities/text
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import daft
import pandas as pd
import psutil
import torch
from daft import DataType, Series
from sentence_transformers import CrossEncoder

# ==================== Resource Monitoring ====================


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
        while self.monitoring:
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)

            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.cpu_per_core_samples.append(cpu_per_core)

            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            time.sleep(self.interval)

    def start(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.cpu_per_core_samples = []

        self.process.cpu_percent(interval=None)

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> Dict:
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

        num_cores = psutil.cpu_count()
        core_utilization = [0] * num_cores

        for sample in self.cpu_per_core_samples:
            for i, util in enumerate(sample):
                core_utilization[i] += util

        if self.cpu_per_core_samples:
            core_utilization = [
                u / len(self.cpu_per_core_samples) for u in core_utilization
            ]

        cores_utilized = sum(1 for u in core_utilization if u > 10)

        return {
            "cpu_mean": sum(self.cpu_samples) / len(self.cpu_samples),
            "cpu_max": max(self.cpu_samples),
            "cpu_cores_utilized": cores_utilized,
            "memory_mean_mb": sum(self.memory_samples) / len(self.memory_samples),
            "memory_max_mb": max(self.memory_samples),
            "num_cores": num_cores,
        }


# ==================== Optimized Rerankers (Daft Best Practices) ====================


# 1. Baseline (Current approach)
@daft.cls
class BaselineReranker:
    """Baseline reranker without optimizations."""

    def __init__(self, model_name: str, batch_size: int = 32):
        print(f"[Baseline] Loading {model_name}...")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()
        pairs = [[q, c] for q, c in zip(queries, candidates)]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return scores.tolist()


# 2. Daft-style optimization: GPU + Compilation + Process isolation
@daft.cls(max_concurrency=1, use_process=True)
class DaftOptimizedReranker:
    """Reranker with Daft best practices: GPU support, compilation, process isolation."""

    def __init__(self, model_name: str, batch_size: int = 32):
        print(f"[Daft-Optimized] Loading {model_name}...")

        # Detect device (following Daft's pattern)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {self.device}")

        # Load model to device
        self.model = CrossEncoder(model_name, device=self.device)

        # Compile model for optimization (Daft pattern from their examples)
        if hasattr(self.model, "model"):
            print("  Compiling model with torch.compile()...")
            try:
                self.model.model = torch.compile(self.model.model)
                print("  ✓ Model compiled")
            except Exception as e:
                print(f"  ✗ Compilation failed: {e}")

        self.batch_size = batch_size

    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()
        pairs = [[q, c] for q, c in zip(queries, candidates)]

        # Use inference mode for efficiency (Daft pattern)
        with torch.inference_mode():
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )

        return scores.tolist()


# 3. Daft-style with larger batch (following their embedding examples)
@daft.cls(max_concurrency=1, use_process=True)
class DaftLargeBatchReranker:
    """Reranker with larger batch size (Daft embedding pattern)."""

    def __init__(self, model_name: str, batch_size: int = 128):
        print(f"[Daft-LargeBatch] Loading {model_name} (batch_size={batch_size})...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {self.device}")

        self.model = CrossEncoder(model_name, device=self.device)

        if hasattr(self.model, "model"):
            try:
                self.model.model = torch.compile(self.model.model)
                print("  ✓ Model compiled")
            except:
                pass

        self.batch_size = batch_size

    @daft.method.batch(return_dtype=DataType.float64(), batch_size=128)
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()
        pairs = [[q, c] for q, c in zip(queries, candidates)]

        with torch.inference_mode():
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )

        return scores.tolist()


# 4. Simple threading optimization (no Daft UDF changes)
@daft.cls
class ThreadedReranker:
    """Reranker with threading optimization only."""

    def __init__(self, model_name: str, batch_size: int = 32):
        import os

        # Set threading BEFORE loading model
        num_threads = psutil.cpu_count()
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        print(f"[Threaded] Loading {model_name} (threads={num_threads})...")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()
        pairs = [[q, c] for q, c in zip(queries, candidates)]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return scores.tolist()


# ==================== Data & Benchmark ====================


def load_benchmark_data(n_queries: int = 10, n_candidates_per_query: int = 30):
    """Load queries and candidates for reranking benchmark."""
    queries_df = pd.read_parquet("data/sample_10/dev.parquet")
    query_df = (
        queries_df[["query_uuid", "query_text"]].drop_duplicates().head(n_queries)
    )

    corpus_df = pd.read_parquet("data/sample_10/corpus.parquet")

    flat_data = []

    for _, query_row in query_df.iterrows():
        query_uuid = query_row["query_uuid"]
        query_text = query_row["query_text"]

        candidates = corpus_df.sample(n=min(n_candidates_per_query, len(corpus_df)))
        candidate_texts = candidates["passage"].tolist()
        candidate_uuids = candidates["uuid"].tolist()

        for cand_text, cand_uuid in zip(candidate_texts, candidate_uuids):
            flat_data.append(
                {
                    "query_uuid": query_uuid,
                    "query_text": query_text,
                    "candidate_text": cand_text,
                    "candidate_uuid": cand_uuid,
                }
            )

    print(f"Loaded {n_queries} queries with ~{n_candidates_per_query} candidates each")
    print(f"Total query-candidate pairs: {len(flat_data)}")

    return flat_data


def benchmark_strategy(
    strategy_name: str,
    df: daft.DataFrame,
    reranker_instance,
    total_pairs: int,
):
    """Benchmark a reranking strategy."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {strategy_name}")
    print(f"{'=' * 70}")

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    start = time.perf_counter()

    result_df = df.with_column(
        "score",
        reranker_instance.score_pairs(df["query_text"], df["candidate_text"]),
    )

    result = result_df.collect()

    elapsed = time.perf_counter() - start

    resource_stats = monitor.stop()

    pairs_per_sec = total_pairs / elapsed if elapsed > 0 else 0

    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Throughput: {pairs_per_sec:.1f} pairs/sec")
    print(
        f"  CPU: {resource_stats['cpu_mean']:.1f}% avg, {resource_stats['cpu_max']:.1f}% peak"
    )
    print(
        f"  Cores: {resource_stats['cpu_cores_utilized']}/{resource_stats['num_cores']}"
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
    """Print results table."""
    print("\n" + "=" * 100)
    print("DAFT OPTIMIZATION RESULTS")
    print("=" * 100)

    baseline_time = next(
        (r["time"] for r in results if "Baseline" in r["strategy"]),
        results[0]["time"],
    )

    print(f"\n{'Strategy':<45} {'Time (s)':>10} {'Speedup':>10} {'Pairs/sec':>12}")
    print("-" * 100)

    for result in results:
        speedup = baseline_time / result["time"] if result["time"] > 0 else 0
        marker = (
            "🏆" if result == max(results, key=lambda x: x["pairs_per_sec"]) else "  "
        )
        print(
            f"{marker} {result['strategy']:<43} {result['time']:>10.3f} {speedup:>9.1f}x {result['pairs_per_sec']:>12.1f}"
        )

    print("\n" + "=" * 100)
    print("RESOURCE UTILIZATION")
    print("=" * 100)
    print(
        f"\n{'Strategy':<45} {'CPU Avg':>10} {'CPU Peak':>10} {'Cores':>8} {'Mem':>10}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"  {result['strategy']:<43} "
            f"{result['cpu_mean']:>9.1f}% "
            f"{result['cpu_max']:>9.1f}% "
            f"{result['cores_utilized']:>3}/{result['num_cores']:<3} "
            f"{result['memory_mean_mb']:>8.0f}MB"
        )

    print("=" * 100)

    best = max(results, key=lambda x: x["pairs_per_sec"])
    print(f"\n🏆 WINNER: {best['strategy']}")
    print(
        f"   {best['pairs_per_sec']:.1f} pairs/sec ({baseline_time / best['time']:.1f}x faster than baseline)"
    )
    print(
        f"   CPU: {best['cpu_mean']:.1f}% avg, Cores: {best['cores_utilized']}/{best['num_cores']}"
    )

    print("\n" + "=" * 100)


# ==================== Main ====================


def main():
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_QUERIES = 10
    N_CANDIDATES = 30

    print("=" * 100)
    print("CROSSENCODER RERANKING - DAFT BEST PRACTICES BENCHMARK")
    print("=" * 100)
    print(f"Model: {MODEL_NAME}")
    print(f"Queries: {N_QUERIES}")
    print(f"Candidates per query: {N_CANDIDATES}")
    print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    # Load data
    print("\nLoading benchmark data...")
    flat_data = load_benchmark_data(
        n_queries=N_QUERIES, n_candidates_per_query=N_CANDIDATES
    )
    total_pairs = len(flat_data)

    df_flat = daft.from_pydict(
        {
            "query_uuid": [d["query_uuid"] for d in flat_data],
            "query_text": [d["query_text"] for d in flat_data],
            "candidate_text": [d["candidate_text"] for d in flat_data],
            "candidate_uuid": [d["candidate_uuid"] for d in flat_data],
        }
    )

    results = []

    # 1. Baseline
    print("\n" + ">" * 100)
    print("STRATEGY 1: Baseline (no optimizations)")
    print(">" * 100)
    baseline = BaselineReranker(MODEL_NAME, batch_size=32)
    results.append(
        benchmark_strategy(
            "1. Baseline (no optimizations)", df_flat, baseline, total_pairs
        )
    )

    # 2. Threaded (simple win)
    print("\n" + ">" * 100)
    print("STRATEGY 2: Threading Optimization")
    print(">" * 100)
    threaded = ThreadedReranker(MODEL_NAME, batch_size=32)
    results.append(
        benchmark_strategy("2. Threading Optimization", df_flat, threaded, total_pairs)
    )

    # 3. Daft-optimized (GPU + compilation + process isolation)
    print("\n" + ">" * 100)
    print("STRATEGY 3: Daft-Optimized (GPU + Compilation + Process Isolation)")
    print(">" * 100)
    daft_opt = DaftOptimizedReranker(MODEL_NAME, batch_size=32)
    results.append(
        benchmark_strategy(
            "3. Daft-Optimized (GPU + compile + process)",
            df_flat,
            daft_opt,
            total_pairs,
        )
    )

    # 4. Daft-optimized with larger batch
    print("\n" + ">" * 100)
    print("STRATEGY 4: Daft-Optimized + Large Batch (batch_size=128)")
    print(">" * 100)
    daft_large = DaftLargeBatchReranker(MODEL_NAME, batch_size=128)
    results.append(
        benchmark_strategy(
            "4. Daft-Optimized + Large Batch (128)", df_flat, daft_large, total_pairs
        )
    )

    # Print results
    print_results_table(results)

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    best = max(results, key=lambda x: x["pairs_per_sec"])

    print(f"\n✓ Use: {best['strategy']}")
    print(f"  Expected throughput: {best['pairs_per_sec']:.1f} pairs/sec")
    print(f"  Speedup over baseline: {results[0]['time'] / best['time']:.1f}x")

    if torch.cuda.is_available():
        print("\n💡 GPU detected - utilizing GPU acceleration")
    else:
        print("\n💡 CPU-only mode - consider GPU for 5-10x speedup")

    print("\n💡 Daft Best Practices Applied:")
    print("  - torch.inference_mode() for efficient inference")
    print("  - torch.compile() for JIT optimization")
    print("  - use_process=True for process isolation")
    print("  - max_concurrency=1 to avoid overhead")
    print("  - Automatic GPU detection and usage")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CrossEncoder Reranking - Testing ADVANCED Optimizations
=========================================================

Tests multiple optimization backends and configurations:
1. PyTorch (baseline)
2. PyTorch + Threading optimization
3. Batch size tuning
4. OpenVINO backend (if available)
5. Fast CrossEncoder auto device selection

The benchmark reports throughput and resource usage for each configuration.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List

import daft
import pandas as pd
import psutil
from daft import DataType, Series
from fast_cross_encoder import FastCrossEncoderScorer
from sentence_transformers import CrossEncoder


def _read_int_env(var_name: str, default: int) -> int:
    """Read an integer value from the environment with a safe fallback."""
    value = os.environ.get(var_name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


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
        """Background monitoring thread."""
        while self.monitoring:
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)

            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.cpu_per_core_samples.append(cpu_per_core)

            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.cpu_per_core_samples = []

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


# ==================== Data Loading ====================


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


# ==================== Optimized Rerankers ====================


# 1. PyTorch Baseline
@daft.cls
class PyTorchReranker:
    """Baseline PyTorch reranker."""

    def __init__(self, model_name: str, batch_size: int = 32):
        print(f"[PyTorch] Loading {model_name} (batch_size={batch_size})...")
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


# 2. PyTorch with Optimized Threading
@daft.cls
class PyTorchOptimizedReranker:
    """PyTorch with optimized threading settings."""

    def __init__(self, model_name: str, batch_size: int = 32, num_threads: int = None):
        import torch

        # Set threading before model load
        if num_threads is None:
            num_threads = psutil.cpu_count()

        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        print(
            f"[PyTorch+Threading] Loading {model_name} (threads={num_threads}, batch_size={batch_size})..."
        )
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


# 4. OpenVINO Backend (if available)
@daft.cls
class OpenVINOReranker:
    """OpenVINO-optimized reranker (optimized for Intel CPUs)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        print(f"[OpenVINO] Loading {model_name} (batch_size={batch_size})...")
        try:
            self.model = CrossEncoder(model_name, backend="openvino")
            print("  ✓ OpenVINO backend loaded successfully")
        except Exception as e:
            print(f"  ✗ OpenVINO backend failed: {e}")
            print("  → Falling back to PyTorch")
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


@daft.cls
class FastCrossEncoderReranker:
    """Wrapper around FastCrossEncoderScorer for Daft benchmarking."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 256,
        warmup: bool = True,
    ):
        print(
            f"[FastCrossEncoder] Loading {model_name} "
            f"(batch_size={batch_size}, warmup={warmup})..."
        )
        self.scorer = FastCrossEncoderScorer(
            model_name,
            batch_size=batch_size,
            warmup=warmup,
        )
        print(f"  ✓ Device: {self.scorer.resolved_device}")

    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        return self.scorer.score_pairs(
            query_texts.to_pylist(),
            candidate_texts.to_pylist(),
        )


# ==================== Benchmark Function ====================


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
    """Print comprehensive results table."""
    print("\n" + "=" * 100)
    print("OPTIMIZATION RESULTS")
    print("=" * 100)

    baseline_time = next(
        (r["time"] for r in results if "PyTorch Baseline" in r["strategy"]),
        results[0]["time"],
    )

    # Performance table
    print(f"\n{'Strategy':<40} {'Time (s)':>10} {'Speedup':>10} {'Pairs/sec':>12}")
    print("-" * 100)

    for result in results:
        speedup = baseline_time / result["time"] if result["time"] > 0 else 0
        marker = (
            "🏆" if result == max(results, key=lambda x: x["pairs_per_sec"]) else "  "
        )
        print(
            f"{marker} {result['strategy']:<38} {result['time']:>10.3f} {speedup:>9.1f}x {result['pairs_per_sec']:>12.1f}"
        )

    # Resource table
    print("\n" + "=" * 100)
    print("RESOURCE UTILIZATION")
    print("=" * 100)
    print(
        f"\n{'Strategy':<40} {'CPU Avg':>10} {'CPU Peak':>10} {'Cores':>8} {'Mem':>10}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"  {result['strategy']:<38} "
            f"{result['cpu_mean']:>9.1f}% "
            f"{result['cpu_max']:>9.1f}% "
            f"{result['cores_utilized']:>3}/{result['num_cores']:<3} "
            f"{result['memory_mean_mb']:>8.0f}MB"
        )

    print("=" * 100)

    # Winner
    best = max(results, key=lambda x: x["pairs_per_sec"])
    print(f"\n🏆 WINNER: {best['strategy']}")
    print(
        f"   {best['pairs_per_sec']:.1f} pairs/sec ({baseline_time / best['time']:.1f}x faster than baseline)"
    )
    print(
        f"   CPU: {best['cpu_mean']:.1f}% avg, Cores: {best['cores_utilized']}/{best['num_cores']}"
    )

    print("\n" + "=" * 100)


def run_benchmark(
    model_name: str,
    n_queries: int,
    n_candidates: int,
) -> List[Dict]:
    """Run the Daft-based benchmark and return collected results."""
    print("\nLoading benchmark data...")
    flat_data = load_benchmark_data(
        n_queries=n_queries,
        n_candidates_per_query=n_candidates,
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

    results: List[Dict] = []

    print("\n" + ">" * 100)
    print("STRATEGY 1: PyTorch Baseline (batch_size=32)")
    print(">" * 100)
    pytorch_baseline = PyTorchReranker(model_name, batch_size=32)
    results.append(
        benchmark_strategy(
            "1. PyTorch Baseline (batch_size=32)",
            df_flat,
            pytorch_baseline,
            total_pairs,
        )
    )

    print("\n" + ">" * 100)
    print("STRATEGY 2: PyTorch + Optimized Threading")
    print(">" * 100)
    pytorch_threaded = PyTorchOptimizedReranker(
        model_name, batch_size=32, num_threads=psutil.cpu_count()
    )
    results.append(
        benchmark_strategy(
            "2. PyTorch + Threading (all cores)",
            df_flat,
            pytorch_threaded,
            total_pairs,
        )
    )

    print("\n" + ">" * 100)
    print("STRATEGY 3: PyTorch (batch_size=64)")
    print(">" * 100)
    pytorch_large_batch = PyTorchReranker(model_name, batch_size=64)
    results.append(
        benchmark_strategy(
            "3. PyTorch (batch_size=64)",
            df_flat,
            pytorch_large_batch,
            total_pairs,
        )
    )

    print("\n" + ">" * 100)
    print("STRATEGY 4: OpenVINO Backend (Intel CPU optimized)")
    print(">" * 100)
    try:
        openvino_reranker = OpenVINOReranker(model_name, batch_size=32)
        results.append(
            benchmark_strategy(
                "4. OpenVINO Backend (batch_size=32)",
                df_flat,
                openvino_reranker,
                total_pairs,
            )
        )
    except Exception as e:
        print(f"✗ OpenVINO backend unavailable: {e}")
        print("  → Install with: pip install optimum[openvino]")

    print("\n" + ">" * 100)
    print("STRATEGY 5: Fast CrossEncoder (auto device selection)")
    print(">" * 100)
    fast_reranker = FastCrossEncoderReranker(
        model_name,
        batch_size=128,
        warmup=True,
    )
    results.append(
        benchmark_strategy(
            "5. Fast CrossEncoder (batch_size=128)",
            df_flat,
            fast_reranker,
            total_pairs,
        )
    )

    print_results_table(results)

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR PRODUCTION")
    print("=" * 100)

    best = max(results, key=lambda x: x["pairs_per_sec"])

    print(f"\n✓ Use: {best['strategy']}")
    print(f"  Expected throughput: {best['pairs_per_sec']:.1f} pairs/sec")
    print(f"  Speedup over baseline: {results[0]['time'] / best['time']:.1f}x")

    if "OpenVINO" in best["strategy"]:
        print("\n💡 OpenVINO Installation:")
        print("  pip install optimum[openvino]")
        print("\n💡 Usage:")
        print(f"  model = CrossEncoder('{model_name}', backend='openvino')")
    elif "Fast CrossEncoder" in best["strategy"]:
        print("\n💡 FastCrossEncoder tips:")
        print("  - Auto-selects CUDA → MPS → CPU with warmup enabled")
        print("  - Override env via HN_BENCHMARK_* for safe benchmarking")

    print("\n" + "=" * 100)

    return results


# ==================== Main ====================


def main():
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_QUERIES = _read_int_env("HN_BENCHMARK_QUERIES", 10)
    N_CANDIDATES = _read_int_env("HN_BENCHMARK_CANDIDATES", 30)

    print("=" * 100)
    if os.environ.get("HN_BENCHMARK_QUERIES") or os.environ.get(
        "HN_BENCHMARK_CANDIDATES"
    ):
        print(
            "💡 Running in gentle mode via HN_BENCHMARK_* environment overrides "
            f"({N_QUERIES} queries, {N_CANDIDATES} candidates)."
        )
    print("CROSSENCODER RERANKING - ADVANCED OPTIMIZATIONS BENCHMARK")
    print("=" * 100)
    print(f"Model: {MODEL_NAME}")
    print(f"Queries: {N_QUERIES}")
    print(f"Candidates per query: {N_CANDIDATES}")
    print("=" * 100)

    run_benchmark(MODEL_NAME, N_QUERIES, N_CANDIDATES)


if __name__ == "__main__":
    main()

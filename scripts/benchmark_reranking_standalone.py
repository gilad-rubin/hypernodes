#!/usr/bin/env python3
"""
Standalone CrossEncoder reranking benchmark (no Daft dependency).
================================================================

Runs several optimized configurations directly in Python to measure
throughput and resource usage when scoring query–candidate pairs.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import psutil
import torch
from sentence_transformers import CrossEncoder

from fast_cross_encoder import FastCrossEncoderScorer

Pair = Tuple[str, str]


# ==================== Resource Monitoring ====================


class ResourceMonitor:
    """Monitor CPU and memory usage during execution."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.cpu_per_core_samples: List[Sequence[float]] = []
        self.thread = None
        self.process = psutil.Process()

    def _monitor(self) -> None:
        while self.monitoring:
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)

            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.cpu_per_core_samples.append(cpu_per_core)

            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            time.sleep(self.interval)

    def start(self) -> None:
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
                "num_cores": psutil.cpu_count(),
            }

        num_cores = psutil.cpu_count()
        core_utilization = [0.0] * num_cores

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


def load_benchmark_pairs(
    n_queries: int = 10,
    n_candidates_per_query: int = 30,
) -> List[Pair]:
    """Load benchmark dataset and return flat query/candidate pairs."""
    queries_df = pd.read_parquet("data/sample_10/dev.parquet")
    query_df = (
        queries_df[["query_uuid", "query_text"]]
        .drop_duplicates()
        .head(n_queries)
    )

    corpus_df = pd.read_parquet("data/sample_10/corpus.parquet")

    pairs: List[Pair] = []

    for _, query_row in query_df.iterrows():
        query_text = query_row["query_text"]
        candidates = corpus_df.sample(n=min(n_candidates_per_query, len(corpus_df)))
        candidate_texts = candidates["passage"].tolist()
        pairs.extend((query_text, cand_text) for cand_text in candidate_texts)

    print(f"Loaded {n_queries} queries with ~{n_candidates_per_query} candidates each")
    print(f"Total query-candidate pairs: {len(pairs)}")
    return pairs


# ==================== Rerankers ====================


class PyTorchReranker:
    """Baseline PyTorch CrossEncoder reranker."""

    def __init__(self, model_name: str, batch_size: int = 32):
        print(f"[PyTorch] Loading {model_name} (batch_size={batch_size})...")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def score_pairs(self, pairs: Sequence[Pair]) -> List[float]:
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        ).tolist()


class PyTorchThreadedReranker(PyTorchReranker):
    """PyTorch reranker with explicit threading configuration."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        num_threads: int | None = None,
    ):
        if num_threads is None:
            num_threads = psutil.cpu_count()

        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        print(
            f"[PyTorch+Threading] Loading {model_name} "
            f"(threads={num_threads}, batch_size={batch_size})..."
        )
        super().__init__(model_name, batch_size=batch_size)


class FastCrossEncoderReranker:
    """Wrapper around FastCrossEncoderScorer."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 128,
        warmup: bool = True,
    ):
        print(
            f"[FastCrossEncoder] Loading {model_name} "
            f"(batch_size={batch_size}, warmup={warmup})..."
        )
        self.scorer = FastCrossEncoderScorer(
            model_name=model_name,
            batch_size=batch_size,
            warmup=warmup,
        )
        print(f"  ✓ Device: {self.scorer.resolved_device}")

    def score_pairs(self, pairs: Sequence[Pair]) -> List[float]:
        queries, candidates = zip(*pairs)
        return self.scorer.score_pairs(list(queries), list(candidates))


# ==================== Benchmarking ====================


def benchmark_strategy(
    strategy_name: str,
    reranker,
    pairs: Sequence[Pair],
) -> Dict:
    """Benchmark a reranking strategy."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {strategy_name}")
    print(f"{'=' * 70}")

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    start = time.perf_counter()
    scores = reranker.score_pairs(pairs)
    elapsed = time.perf_counter() - start

    monitor_stats = monitor.stop()

    pairs_per_sec = len(scores) / elapsed if elapsed > 0 else 0

    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Throughput: {pairs_per_sec:.1f} pairs/sec")
    print(
        f"  CPU: {monitor_stats['cpu_mean']:.1f}% avg, "
        f"{monitor_stats['cpu_max']:.1f}% peak"
    )
    print(
        f"  Cores: {monitor_stats['cpu_cores_utilized']}/"
        f"{monitor_stats['num_cores']}"
    )
    print(
        f"  Memory: {monitor_stats['memory_mean_mb']:.1f} MB avg, "
        f"{monitor_stats['memory_max_mb']:.1f} MB peak"
    )

    return {
        "strategy": strategy_name,
        "time": elapsed,
        "pairs_per_sec": pairs_per_sec,
        "cpu_mean": monitor_stats["cpu_mean"],
        "cpu_max": monitor_stats["cpu_max"],
        "cores_utilized": monitor_stats["cpu_cores_utilized"],
        "num_cores": monitor_stats["num_cores"],
        "memory_mean_mb": monitor_stats["memory_mean_mb"],
        "memory_max_mb": monitor_stats["memory_max_mb"],
    }


def print_results_table(results: List[Dict]) -> None:
    """Print summary tables."""
    print("\n" + "=" * 100)
    print("STANDALONE OPTIMIZATION RESULTS")
    print("=" * 100)

    baseline_time = results[0]["time"]

    print(f"\n{'Strategy':<40} {'Time (s)':>10} {'Speedup':>10} {'Pairs/sec':>12}")
    print("-" * 100)
    best = max(results, key=lambda x: x["pairs_per_sec"])

    for result in results:
        speedup = baseline_time / result["time"] if result["time"] > 0 else 0
        marker = "🏆 " if result is best else "   "
        print(
            f"{marker}{result['strategy']:<38} "
            f"{result['time']:>10.3f} "
            f"{speedup:>9.1f}x "
            f"{result['pairs_per_sec']:>12.1f}"
        )

    print("\n" + "=" * 100)
    print("RESOURCE UTILIZATION")
    print("=" * 100)
    print(
        f"\n{'Strategy':<40} {'CPU Avg':>10} {'CPU Peak':>10} "
        f"{'Cores':>8} {'Mem':>10}"
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
    print(f"\n🏆 WINNER: {best['strategy']}")
    print(
        f"   {best['pairs_per_sec']:.1f} pairs/sec "
        f"({baseline_time / best['time']:.1f}x faster than baseline)"
    )
    print(
        f"   CPU: {best['cpu_mean']:.1f}% avg, "
        f"Cores: {best['cores_utilized']}/{best['num_cores']}"
    )
    print("\n" + "=" * 100)


# ==================== Main ====================


def _read_int_env(var_name: str, default: int) -> int:
    value = os.environ.get(var_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def run_benchmark(
    model_name: str,
    n_queries: int,
    n_candidates: int,
) -> List[Dict]:
    """Run the standalone benchmark and return collected results."""
    print("\nLoading benchmark data...")
    pairs = load_benchmark_pairs(
        n_queries=n_queries,
        n_candidates_per_query=n_candidates,
    )

    strategies = [
        (
            "1. PyTorch Baseline (batch_size=32)",
            PyTorchReranker(model_name, batch_size=32),
        ),
        (
            "2. PyTorch + Threading (all cores)",
            PyTorchThreadedReranker(
                model_name,
                batch_size=32,
                num_threads=psutil.cpu_count(),
            ),
        ),
        (
            "3. PyTorch (batch_size=64)",
            PyTorchReranker(model_name, batch_size=64),
        ),
        (
            "4. Fast CrossEncoder (batch_size=128)",
            FastCrossEncoderReranker(
                model_name,
                batch_size=128,
                warmup=True,
            ),
        ),
    ]

    results = [benchmark_strategy(name, reranker, pairs) for name, reranker in strategies]

    print_results_table(results)

    return results


def main() -> None:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_QUERIES = _read_int_env("HN_BENCHMARK_QUERIES", 10)
    N_CANDIDATES = _read_int_env("HN_BENCHMARK_CANDIDATES", 30)

    print("=" * 100)
    if (
        os.environ.get("HN_BENCHMARK_QUERIES")
        or os.environ.get("HN_BENCHMARK_CANDIDATES")
    ):
        print(
            "💡 Running in gentle mode via HN_BENCHMARK_* environment overrides "
            f"({N_QUERIES} queries, {N_CANDIDATES} candidates)."
        )
    print("STANDALONE CROSSENCODER RERANKING BENCHMARK")
    print("=" * 100)
    print(f"Model: {MODEL_NAME}")
    print(f"Queries: {N_QUERIES}")
    print(f"Candidates per query: {N_CANDIDATES}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print("=" * 100)

    run_benchmark(MODEL_NAME, N_QUERIES, N_CANDIDATES)


if __name__ == "__main__":
    main()


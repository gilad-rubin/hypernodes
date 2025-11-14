#!/usr/bin/env python3
"""
Compare Daft-based vs. standalone CrossEncoder benchmarks.
Runs both benchmarks with the same parameters and summarizes the winner.
"""

from __future__ import annotations

import os
from typing import Dict, List

from benchmark_reranking_optimized import run_benchmark as run_daft_benchmark
from benchmark_reranking_standalone import run_benchmark as run_standalone_benchmark


def _read_int_env(var_name: str, default: int) -> int:
    value = os.environ.get(var_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _best_result(results: List[Dict]) -> Dict:
    return max(results, key=lambda x: x["pairs_per_sec"])


def _print_comparison_table(daft_best: Dict, standalone_best: Dict) -> None:
    print("\n" + "=" * 80)
    print("DAFT VS. STANDALONE SUMMARY")
    print("=" * 80)
    print(f"\n{'Pipeline':<20} {'Best Strategy':<40} {'Pairs/sec':>12} {'Time (s)':>10}")
    print("-" * 80)
    print(
        f"{'Daft':<20} {daft_best['strategy']:<40} "
        f"{daft_best['pairs_per_sec']:>12.1f} {daft_best['time']:>10.3f}"
    )
    print(
        f"{'Standalone':<20} {standalone_best['strategy']:<40} "
        f"{standalone_best['pairs_per_sec']:>12.1f} {standalone_best['time']:>10.3f}"
    )

    winner = (
        ("Daft", daft_best)
        if daft_best["pairs_per_sec"] >= standalone_best["pairs_per_sec"]
        else ("Standalone", standalone_best)
    )
    loser = standalone_best if winner[0] == "Daft" else daft_best
    speedup = (
        winner[1]["pairs_per_sec"] / loser["pairs_per_sec"]
        if loser["pairs_per_sec"] > 0
        else float("inf")
    )

    print("\n" + "=" * 80)
    print(f"🏆 OVERALL WINNER: {winner[0]} Path")
    print(
        f"   {winner[1]['pairs_per_sec']:.1f} pairs/sec "
        f"({speedup:.2f}x faster than the other pipeline)"
    )
    print("=" * 80)


def main() -> None:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    N_QUERIES = _read_int_env("HN_BENCHMARK_QUERIES", 20)
    N_CANDIDATES = _read_int_env("HN_BENCHMARK_CANDIDATES", 30)

    print("=" * 100)
    if (
        os.environ.get("HN_BENCHMARK_QUERIES")
        or os.environ.get("HN_BENCHMARK_CANDIDATES")
    ):
        print(
            "💡 Using HN_BENCHMARK_* overrides "
            f"({N_QUERIES} queries, {N_CANDIDATES} candidates)."
        )
    else:
        print(f"Running comparison with {N_QUERIES} queries, {N_CANDIDATES} candidates.")
    print("COMPARE: Daft vs. Standalone CrossEncoder Benchmarks")
    print("=" * 100)

    print("\n" + "#" * 100)
    print("Running Daft-based benchmark...")
    print("#" * 100)
    daft_results = run_daft_benchmark(MODEL_NAME, N_QUERIES, N_CANDIDATES)

    print("\n" + "#" * 100)
    print("Running Standalone benchmark...")
    print("#" * 100)
    standalone_results = run_standalone_benchmark(
        MODEL_NAME,
        N_QUERIES,
        N_CANDIDATES,
    )

    daft_best = _best_result(daft_results)
    standalone_best = _best_result(standalone_results)
    _print_comparison_table(daft_best, standalone_best)


if __name__ == "__main__":
    main()


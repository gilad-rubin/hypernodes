#!/usr/bin/env python3
"""
Simple, fast benchmark: Custom vs Daft-Optimized cross-encoder.
"""

import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fast_cross_encoder import FastCrossEncoderScorer


def load_data(n_queries: int = 10, n_candidates: int = 30):
    """Load benchmark data."""
    queries_df = pd.read_parquet("data/sample_10/dev.parquet")
    query_df = queries_df[["query_uuid", "query_text"]].drop_duplicates().head(n_queries)
    corpus_df = pd.read_parquet("data/sample_10/corpus.parquet")

    flat_data = []
    for _, query_row in query_df.iterrows():
        candidates = corpus_df.sample(n=min(n_candidates, len(corpus_df)))
        for _, cand_row in candidates.iterrows():
            flat_data.append({
                "query_text": query_row["query_text"],
                "candidate_text": cand_row["passage"],
            })
    
    return flat_data


def benchmark_custom(data: List[dict], batch_sizes: List[int]):
    """Benchmark custom implementation with different batch sizes."""
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    print("=" * 80)
    print("CUSTOM FAST CROSS-ENCODER BENCHMARK")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Total pairs: {len(data)}")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'─' * 80}")
        print(f"Testing batch_size={batch_size}")
        print(f"{'─' * 80}")
        
        # Initialize scorer
        scorer = FastCrossEncoderScorer(MODEL_NAME, batch_size=batch_size)
        print(f"Device: {scorer.device}")
        
        # Extract data
        queries = [d["query_text"] for d in data]
        candidates = [d["candidate_text"] for d in data]
        
        # Benchmark
        start = time.perf_counter()
        scores = scorer.score_pairs(queries, candidates)
        elapsed = time.perf_counter() - start
        
        throughput = len(data) / elapsed
        
        print(f"\n✓ Completed in {elapsed:.3f}s")
        print(f"  Throughput: {throughput:.1f} pairs/sec")
        print(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
        
        results.append({
            "batch_size": batch_size,
            "time": elapsed,
            "throughput": throughput,
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Batch Size':>12} {'Time (s)':>12} {'Throughput':>15} {'Speedup':>10}")
    print("─" * 80)
    
    baseline = results[0]
    for r in results:
        speedup = baseline["time"] / r["time"]
        marker = "🏆" if r == max(results, key=lambda x: x["throughput"]) else "  "
        print(
            f"{marker} {r['batch_size']:>10} {r['time']:>12.3f} {r['throughput']:>12.1f} p/s {speedup:>9.1f}x"
        )
    
    best = max(results, key=lambda x: x["throughput"])
    print(f"\n🏆 Best: batch_size={best['batch_size']}, {best['throughput']:.1f} pairs/sec")
    print("=" * 80)


def main():
    # Load data
    print("Loading data...")
    data = load_data(n_queries=10, n_candidates=30)
    print(f"Loaded {len(data)} pairs\n")
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    benchmark_custom(data, batch_sizes)


if __name__ == "__main__":
    main()




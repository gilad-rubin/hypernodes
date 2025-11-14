#!/usr/bin/env python3
"""
Final benchmark: Custom Fast Cross-Encoder vs Daft-Optimized.
Safe version - loads each model only once.
"""

import time
from typing import List
import pandas as pd
import torch

# Custom implementation
from sentence_transformers import CrossEncoder


class FastCrossEncoderScorer:
    """Optimized cross-encoder scorer."""
    
    def __init__(self, model_name: str, batch_size: int = 128):
        self.model = CrossEncoder(model_name)  # Auto-detect device
        self.batch_size = batch_size
        # Warmup
        warmup_pairs = [["warmup", "warmup"]] * min(batch_size, 32)
        self.model.predict(warmup_pairs, batch_size=batch_size, show_progress_bar=False)
    
    def score_pairs(self, query_texts: List[str], candidate_texts: List[str]) -> List[float]:
        pairs = [[q, c] for q, c in zip(query_texts, candidate_texts)]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return scores.tolist()
    
    @property
    def device(self):
        return str(self.model.model.device) if hasattr(self.model, 'model') else "unknown"


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


def benchmark_custom(name: str, data: List[dict], scorer: FastCrossEncoderScorer):
    """Benchmark custom implementation."""
    queries = [d["query_text"] for d in data]
    candidates = [d["candidate_text"] for d in data]
    
    start = time.perf_counter()
    scores = scorer.score_pairs(queries, candidates)
    elapsed = time.perf_counter() - start
    
    throughput = len(data) / elapsed
    
    return {
        "name": name,
        "time": elapsed,
        "throughput": throughput,
        "total_pairs": len(data),
    }


def main():
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    print("=" * 80)
    print("CUSTOM CROSS-ENCODER BENCHMARK")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    
    # Load data
    print("\nLoading data...")
    data = load_data(n_queries=10, n_candidates=30)
    print(f"Loaded {len(data)} query-candidate pairs")
    
    results = []
    
    # Test 1: batch_size=32
    print("\n" + "─" * 80)
    print("Test 1: Custom Fast Cross-Encoder (batch_size=32)")
    print("─" * 80)
    scorer_32 = FastCrossEncoderScorer(MODEL_NAME, batch_size=32)
    print(f"Device: {scorer_32.device}")
    result = benchmark_custom("Custom (batch=32)", data, scorer_32)
    print(f"✓ Time: {result['time']:.3f}s")
    print(f"  Throughput: {result['throughput']:.1f} pairs/sec")
    results.append(result)
    
    # Test 2: batch_size=64
    print("\n" + "─" * 80)
    print("Test 2: Custom Fast Cross-Encoder (batch_size=64)")
    print("─" * 80)
    scorer_64 = FastCrossEncoderScorer(MODEL_NAME, batch_size=64)
    print(f"Device: {scorer_64.device}")
    result = benchmark_custom("Custom (batch=64)", data, scorer_64)
    print(f"✓ Time: {result['time']:.3f}s")
    print(f"  Throughput: {result['throughput']:.1f} pairs/sec")
    results.append(result)
    
    # Test 3: batch_size=128
    print("\n" + "─" * 80)
    print("Test 3: Custom Fast Cross-Encoder (batch_size=128)")
    print("─" * 80)
    scorer_128 = FastCrossEncoderScorer(MODEL_NAME, batch_size=128)
    print(f"Device: {scorer_128.device}")
    result = benchmark_custom("Custom (batch=128)", data, scorer_128)
    print(f"✓ Time: {result['time']:.3f}s")
    print(f"  Throughput: {result['throughput']:.1f} pairs/sec")
    results.append(result)
    
    # Test 4: batch_size=256
    print("\n" + "─" * 80)
    print("Test 4: Custom Fast Cross-Encoder (batch_size=256)")
    print("─" * 80)
    scorer_256 = FastCrossEncoderScorer(MODEL_NAME, batch_size=256)
    print(f"Device: {scorer_256.device}")
    result = benchmark_custom("Custom (batch=256)", data, scorer_256)
    print(f"✓ Time: {result['time']:.3f}s")
    print(f"  Throughput: {result['throughput']:.1f} pairs/sec")
    results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Strategy':<30} {'Time (s)':>12} {'Throughput':>20} {'Speedup':>10}")
    print("─" * 80)
    
    baseline = results[0]
    for r in results:
        speedup = baseline["time"] / r["time"]
        marker = "🏆" if r == max(results, key=lambda x: x["throughput"]) else "  "
        print(
            f"{marker} {r['name']:<28} {r['time']:>12.3f} {r['throughput']:>15.1f} p/s {speedup:>9.1f}x"
        )
    
    best = max(results, key=lambda x: x["throughput"])
    print(f"\n🏆 Best: {best['name']}")
    print(f"   Throughput: {best['throughput']:.1f} pairs/sec")
    print(f"   Speedup: {baseline['time'] / best['time']:.2f}x over batch_size=32")
    
    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("• This is a pure Python implementation (no Daft)")
    print("• Auto-detects best device (GPU/MPS/CPU)")
    print("• Simple API: just load model and call score_pairs()")
    print(f"• Device used: {scorer_128.device}")
    print("=" * 80)


if __name__ == "__main__":
    main()



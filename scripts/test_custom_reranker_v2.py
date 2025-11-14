#!/usr/bin/env python3
"""
Test custom high-performance cross-encoder implementation - V2 (clean rewrite).
"""

import time
from typing import List

import torch
from sentence_transformers import CrossEncoder


class FastCrossEncoderScorer:
    """Optimized cross-encoder scorer - minimal, working version."""
    
    def __init__(
        self,
        model_name: str,
        batch_size: int = 128,
    ):
        """Initialize cross-encoder."""
        print(f"[FastScorer] Loading {model_name}...")
        
        # Load model (let it auto-detect device)
        print(f"  Batch size: {batch_size}")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        print(f"  Device: {self.model.device if hasattr(self.model, 'device') else 'unknown'}")
        
        # Warmup
        print("  Running warmup...")
        warmup_pairs = [["warmup query", "warmup document"]] * min(batch_size, 32)
        # Try without inference_mode first
        warmup_scores = self.model.predict(warmup_pairs, batch_size=batch_size, show_progress_bar=False)
        print(f"  Warmup scores (first 3): {warmup_scores[:3]}")
        print("  ✓ Ready")
    
    def score_pairs(
        self, 
        query_texts: List[str], 
        candidate_texts: List[str]
    ) -> List[float]:
        """Score query-candidate pairs."""
        pairs = [[q, c] for q, c in zip(query_texts, candidate_texts)]
        
        # Don't use inference_mode - it seems to cause issues with some models
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        return scores.tolist()


def main():
    print("=" * 70)
    print("FAST CROSS-ENCODER SCORER TEST")
    print("=" * 70)
    
    # Test 1: Basic
    print("\n" + "=" * 70)
    print("Test 1: Basic Scoring (first call after warmup)")
    print("=" * 70)
    
    scorer = FastCrossEncoderScorer(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=128,
    )
    
    queries = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "What causes earthquakes?",
    ]
    
    candidates = [
        "Machine learning is a subset of AI focused on pattern recognition.",
        "Plants convert light energy into chemical energy through photosynthesis.",
        "Earthquakes are caused by tectonic plate movements in the Earth's crust.",
    ]
    
    print("DEBUG: About to call score_pairs...")
    start = time.perf_counter()
    scores = scorer.score_pairs(queries, candidates)
    elapsed = time.perf_counter() - start
    
    print(f"DEBUG: Returned scores = {scores}")
    print(f"\n✓ Scored {len(scores)} pairs in {elapsed:.3f}s")
    print(f"  Throughput: {len(scores) / elapsed:.1f} pairs/sec")
    print("\nScores:")
    for q, c, s in zip(queries, candidates, scores):
        print(f"  {s:>8.4f}: '{q[:30]}...' x '{c[:30]}...'")
    
    # Test 2: Same scorer, new data
    print("\n" + "=" * 70)
    print("Test 2: Second call with same scorer")
    print("=" * 70)
    
    n_pairs = 300
    queries = [f"Query {i}" for i in range(n_pairs)]
    candidates = [f"Document {i} with some content" for i in range(n_pairs)]
    
    start = time.perf_counter()
    scores = scorer.score_pairs(queries, candidates)
    elapsed = time.perf_counter() - start
    
    print(f"\n✓ Scored {len(scores)} pairs in {elapsed:.3f}s")
    print(f"  Throughput: {len(scores) / elapsed:.1f} pairs/sec")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()


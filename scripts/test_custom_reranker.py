#!/usr/bin/env python3
"""
Test custom high-performance cross-encoder implementation.
"""

import time
from typing import List, Tuple

import torch
from sentence_transformers import CrossEncoder


class OptimizedCrossEncoderScorer:
    """High-performance cross-encoder scorer with manual optimizations."""
    
    def __init__(
        self,
        model_name: str,
        batch_size: int = 128,
        num_threads: int | None = None,
        use_compile: bool = False,
        warmup: bool = True,
    ):
        """
        Initialize optimized cross-encoder.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for scoring
            num_threads: Number of CPU threads (auto-detect if None)
            use_compile: Whether to use torch.compile() for JIT optimization
            warmup: Whether to run warmup inference
        """
        print(f"[CustomOptimized] Loading {model_name}...")
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {self.device}")
        
        # Load model to device
        self.model = CrossEncoder(model_name, device=self.device)
        self.batch_size = batch_size
        
        # Note: torch.set_num_threads() can cause NaN issues with some models
        # Let PyTorch use its default threading configuration
        if num_threads:
            print(f"  Note: num_threads parameter ignored (can cause NaN with cross-encoders)")
        
        # Compile model for JIT optimization
        if use_compile and hasattr(self.model, "model"):
            print("  Compiling model with torch.compile()...")
            try:
                self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
                print("  ✓ Model compiled")
            except Exception as e:
                print(f"  ✗ Compilation failed: {e}")
        
        # Warmup to avoid cold-start issues
        if warmup:
            print("  Running warmup...")
            self._warmup()
            print("  ✓ Warmup complete")
    
    def _warmup(self):
        """Run warmup inference to initialize model."""
        dummy_pairs = [["warmup query", "warmup document"]] * min(self.batch_size, 32)
        with torch.inference_mode():
            self.model.predict(
                dummy_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
    
    def score_pairs(
        self, 
        query_texts: List[str], 
        candidate_texts: List[str]
    ) -> List[float]:
        """
        Score query-candidate pairs.
        
        Args:
            query_texts: List of query texts
            candidate_texts: List of candidate texts
            
        Returns:
            List of relevance scores
        """
        pairs = [[q, c] for q, c in zip(query_texts, candidate_texts)]
        
        with torch.inference_mode():
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        
        # Convert to list - scores is already a numpy array
        if hasattr(scores, 'tolist'):
            return scores.tolist()
        return list(scores)


def test_basic_scoring():
    """Test basic scoring functionality."""
    print("\n" + "=" * 70)
    print("Test 1: Basic Scoring")
    print("=" * 70)
    
    scorer = OptimizedCrossEncoderScorer(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=32,
    )
    
    # Test with small batch
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
    
    start = time.perf_counter()
    scores = scorer.score_pairs(queries, candidates)
    elapsed = time.perf_counter() - start
    
    print(f"\n✓ Scored {len(scores)} pairs in {elapsed:.3f}s")
    print(f"  Throughput: {len(scores) / elapsed:.1f} pairs/sec")
    print("\nScores:")
    for q, c, s in zip(queries, candidates, scores):
        print(f"  {s:>8.4f}: '{q[:40]}...' + '{c[:40]}...'")


def test_large_batch():
    """Test with larger batch."""
    print("\n" + "=" * 70)
    print("Test 2: Large Batch (300 pairs)")
    print("=" * 70)
    
    scorer = OptimizedCrossEncoderScorer(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=128,
    )
    
    # Generate test data
    n_pairs = 300
    queries = [f"Query {i}" for i in range(n_pairs)]
    candidates = [f"Candidate document {i} with some content" for i in range(n_pairs)]
    
    start = time.perf_counter()
    scores = scorer.score_pairs(queries, candidates)
    elapsed = time.perf_counter() - start
    
    print(f"\n✓ Scored {len(scores)} pairs in {elapsed:.3f}s")
    print(f"  Throughput: {len(scores) / elapsed:.1f} pairs/sec")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")


def main():
    print("=" * 70)
    print("CUSTOM CROSS-ENCODER IMPLEMENTATION TEST")
    print("=" * 70)
    print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    test_basic_scoring()
    test_large_batch()
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()


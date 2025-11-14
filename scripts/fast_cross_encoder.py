#!/usr/bin/env python3
"""
High-performance cross-encoder scorer - optimized for maximum throughput.
"""

from typing import List, Optional

import torch
from sentence_transformers import CrossEncoder


class FastCrossEncoderScorer:
    """
    Optimized cross-encoder scorer for maximum throughput.

    Key optimizations:
    - Auto device detection with preference order: CUDA → MPS → CPU
    - Large batch processing
    - Optional warmup for consistent performance
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 256,
        warmup: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize cross-encoder scorer.

        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference (larger = better throughput)
            warmup: Whether to run warmup inference
            device: Explicit device override ("cuda", "mps", "cpu"). Auto if None.
        """
        self.device = device or self._detect_device()
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device)

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run a short warmup pass to stabilize throughput."""
        warmup_pairs = [["warmup query", "warmup document"]] * min(self.batch_size, 32)
        self.model.predict(
            warmup_pairs,
            batch_size=min(self.batch_size, 32),
            show_progress_bar=False,
        )

    @staticmethod
    def _detect_device() -> str:
        """Select the fastest available device."""
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def score_pairs(
        self,
        query_texts: List[str],
        candidate_texts: List[str],
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
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return scores.tolist()

    @property
    def resolved_device(self) -> str:
        """Get the device being used."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "device"):
            return str(self.model.model.device)
        return self.device




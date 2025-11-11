#!/usr/bin/env python3
"""Local test of the Modal failure repro (without Modal).

This runs the same pipeline locally with DaftEngine to verify the fix works.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

# Import from the actual repro script
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

# Import all classes and nodes from the repro script
from test_modal_failure_repro import (
    Passage,
    EncodedPassage,
    Query,
    EncodedQuery,
    GroundTruth,
    SearchHit,
    Prediction,
    ColBERTEncoder,
    PLAIDIndex,
    BM25IndexImpl,
    ColBERTReranker,
    RRFFusion,
    NDCGEvaluator,
    RecallEvaluator,
    retrieval_pipeline,
)

from hypernodes import Pipeline
from hypernodes.engines import DaftEngine


def main():
    # Create instances
    encoder = ColBERTEncoder(model_name="mock")
    rrf = RRFFusion(k=10)
    ndcg = NDCGEvaluator(k=2)
    recall = RecallEvaluator(k_list=[1, 2])

    print("=" * 80)
    print("LOCAL TEST: Checking module names before execution")
    print("=" * 80)
    print(f"Passage.__module__ = {Passage.__module__!r}")
    print(f"Prediction.__module__ = {Prediction.__module__!r}")
    print(f"RecallEvaluator.__module__ = {recall.__class__.__module__!r}")
    print()

    inputs = {
        "corpus_path": "mock_corpus.parquet",
        "examples_path": "mock_examples.parquet",
        "model_name": "mock",
        "trust_remote_code": True,
        "index_folder": "index",
        "index_name": "mock",
        "override": True,
        "top_k": 2,
        "rerank_k": 2,
        "rrf_k": 60,
        "ndcg_k": 2,
        "recall_k_list": [1, 2],
        "encoder": encoder,
        "rrf": rrf,
        "ndcg_evaluator": ndcg,
        "recall_evaluator": recall,
    }

    print("=" * 80)
    print("Running pipeline with DaftEngine (debug=True)")
    print("=" * 80)
    print()

    # Run with DaftEngine locally
    engine = DaftEngine(debug=True)
    pipeline = retrieval_pipeline.with_engine(engine)
    
    try:
        result = pipeline.run(inputs=inputs, output_name="evaluation_results")
        print()
        print("=" * 80)
        print("SUCCESS! Pipeline completed")
        print("=" * 80)
        print("Result:", result)
        print()
        
        # Verify the results
        assert "ndcg" in result
        assert "recall@1" in result
        assert "recall@2" in result
        print("✓ All expected metrics present in result")
        
    except Exception as e:
        print()
        print("=" * 80)
        print("FAILURE!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


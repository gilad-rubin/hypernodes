#!/usr/bin/env python3
"""
Exact reproduction of the failing Hebrew Retrieval Pipeline.

This mimics the EXACT structure from the original error, but with mock data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import modal

# Ensure models.py can be imported
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ==================== Mock Implementation Classes ====================
# These mimic the exact structure of your original classes


class ColBERTEncoder:
    """ColBERT encoder implementation (EXACT copy of structure)."""
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True
    __daft_max_concurrency__ = 2
    __daft_gpus__ = 0  # No GPU for testing

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        # Mock: no actual model

    def encode(self, text: str, is_query: bool = False) -> Any:
        """Encode a single text (mock)."""
        # Return mock embedding
        return [0.1, 0.2, 0.3] if is_query else [0.4, 0.5, 0.6]


class PLAIDIndex:
    """PLAID vector index implementation (EXACT copy of structure)."""
    __daft_hint__ = "@daft.cls"

    def __init__(self, encoded_passages: List[Any], index_folder: str, index_name: str, override: bool = True):
        self._documents = {p["uuid"]: p["embedding"] for p in encoded_passages}

    def search(self, query_embedding: Any, k: int) -> List[dict]:
        """Search for top-k results (mock)."""
        # Return mock results
        results = []
        for i, (doc_id, emb) in enumerate(list(self._documents.items())[:k]):
            results.append({"id": doc_id, "score": 1.0 / (i + 1)})
        return results


class BM25IndexImpl:
    """BM25 index implementation (EXACT copy of structure)."""
    __daft_hint__ = "@daft.cls"

    def __init__(self, passages: List[dict]):
        self._passage_uuids = [p["uuid"] for p in passages]

    def search(self, query_text: str, k: int) -> List[dict]:
        """Search for top-k results (mock)."""
        # Return mock results
        results = []
        for i, uuid in enumerate(self._passage_uuids[:k]):
            results.append({"passage_uuid": uuid, "score": 0.9 - i * 0.1})
        return results


class ColBERTReranker:
    """ColBERT reranker implementation (EXACT copy of structure)."""
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True
    __daft_max_concurrency__ = 2

    def __init__(self, encoder: Any, passage_lookup: dict):
        self._encoder = encoder
        self._passages = passage_lookup

    def rerank(self, query: dict, candidates: List[dict], k: int) -> List[dict]:
        """Rerank candidates using ColBERT (mock)."""
        # Just return top k
        return candidates[:k]


class RRFFusion:
    """Reciprocal Rank Fusion implementation (EXACT copy of structure)."""
    __daft_hint__ = "@daft.cls"

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, results_list: List[List[dict]]) -> List[dict]:
        """Fuse multiple retrieval results using RRF (mock)."""
        rrf_scores = {}
        for results in results_list:
            for rank, hit in enumerate(results, 1):
                hit_id = hit.get("passage_uuid") or hit.get("id")
                rrf_scores[hit_id] = rrf_scores.get(hit_id, 0) + 1 / (self.k + rank)
        
        sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"passage_uuid": uuid, "score": score} for uuid, score in sorted_hits]


class NDCGEvaluator:
    """NDCG evaluation implementation (EXACT copy of structure)."""
    __daft_hint__ = "@daft.cls"

    def __init__(self, k: int):
        self.k = k

    def compute(self, predictions: List[dict], ground_truths: List[dict]) -> float:
        """Compute NDCG@k score (mock)."""
        return 0.85  # Mock score


class RecallEvaluator:
    """Recall evaluation implementation (EXACT copy of structure)."""

    def __init__(self, k_list: List[int]):
        self.k_list = k_list

    def compute(self, predictions: List[dict], ground_truths: List[dict]) -> dict:
        """Compute Recall@k for multiple k values (mock)."""
        return {f"recall@{k}": 0.7 + k * 0.01 for k in self.k_list}


# ==================== Simple Nodes (EXACT copy of structure) ====================

from hypernodes import Pipeline, node


@node(output_name="passages")
def load_passages(corpus_path: str) -> List[dict]:
    """Load passages from corpus."""
    return [
        {"uuid": f"p{i}", "text": f"passage {i}"}
        for i in range(5)
    ]


@node(output_name="queries")
def load_queries(examples_path: str) -> List[dict]:
    """Load queries from examples."""
    return [
        {"uuid": f"q{i}", "text": f"query {i}"}
        for i in range(2)
    ]


@node(output_name="ground_truths")
def load_ground_truths(examples_path: str) -> List[dict]:
    """Load ground truth labels."""
    return [
        {"query_uuid": "q0", "paragraph_uuid": "p0", "label_score": 1},
        {"query_uuid": "q1", "paragraph_uuid": "p1", "label_score": 1},
    ]


@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[dict], index_folder: str, index_name: str, override: bool):
    """Build vector index."""
    return PLAIDIndex(encoded_passages, index_folder, index_name, override)


@node(output_name="bm25_index")
def build_bm25_index(passages: List[dict]):
    """Build BM25 index."""
    return BM25IndexImpl(passages)


@node(output_name="passage_lookup")
def build_passage_lookup(encoded_passages: List[dict]) -> dict:
    """Build passage lookup dictionary."""
    return {p["uuid"]: p for p in encoded_passages}


@node(output_name="encoded_passage")
def encode_passage(passage: dict, encoder: ColBERTEncoder) -> dict:
    """Encode a single passage."""
    embedding = encoder.encode(passage["text"], is_query=False)
    return {"uuid": passage["uuid"], "text": passage["text"], "embedding": embedding}


@node(output_name="encoded_query")
def encode_query(query: dict, encoder: ColBERTEncoder) -> dict:
    """Encode a single query."""
    embedding = encoder.encode(query["text"], is_query=True)
    return {"uuid": query["uuid"], "text": query["text"], "embedding": embedding}


@node(output_name="query")
def extract_query(encoded_query: dict) -> dict:
    """Extract Query from EncodedQuery."""
    return {"uuid": encoded_query["uuid"], "text": encoded_query["text"]}


@node(output_name="colbert_hits")
def retrieve_colbert(encoded_query: dict, vector_index: PLAIDIndex, top_k: int) -> List[dict]:
    """Retrieve from ColBERT index."""
    return vector_index.search(encoded_query["embedding"], k=top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(query: dict, bm25_index: BM25IndexImpl, top_k: int) -> List[dict]:
    """Retrieve from BM25 index."""
    return bm25_index.search(query["text"], k=top_k)


@node(output_name="fused_hits")
def fuse_results(colbert_hits: List[dict], bm25_hits: List[dict], rrf: RRFFusion) -> List[dict]:
    """Fuse ColBERT and BM25 results."""
    return rrf.fuse([colbert_hits, bm25_hits])


@node(output_name="reranked_hits")
def rerank_results(
    query: dict,
    fused_hits: List[dict],
    encoder: ColBERTEncoder,
    passage_lookup: dict,
    rerank_k: int,
) -> List[dict]:
    """Rerank fused candidates."""
    reranker = ColBERTReranker(encoder, passage_lookup)
    return reranker.rerank(query, fused_hits, rerank_k)


@node(output_name="predictions")
def hits_to_predictions(query: dict, reranked_hits: List[dict]) -> List[dict]:
    """Convert hits to predictions."""
    return [
        {"query_uuid": query["uuid"], "paragraph_uuid": hit["passage_uuid"], "score": hit["score"]}
        for hit in reranked_hits
    ]


@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[dict]]) -> List[dict]:
    """Flatten nested predictions from mapped results."""
    return [pred for query_preds in all_query_predictions for pred in query_preds]


@node(output_name="ndcg_score")
def compute_ndcg(all_predictions: List[dict], ground_truths: List[dict], ndcg_evaluator: NDCGEvaluator) -> float:
    """Compute NDCG score."""
    return ndcg_evaluator.compute(all_predictions, ground_truths)


@node(output_name="recall_metrics")
def compute_recall(all_predictions: List[dict], ground_truths: List[dict], recall_evaluator: RecallEvaluator) -> dict:
    """Compute Recall metrics."""
    return recall_evaluator.compute(all_predictions, ground_truths)


@node(output_name="evaluation_results")
def combine_evaluation_results(ndcg_score: float, recall_metrics: dict, ndcg_k: int) -> dict:
    """Combine evaluation results into final dict."""
    return {
        "ndcg": ndcg_score,
        "ndcg_k": ndcg_k,
        "recall_metrics": recall_metrics,
    }


# ==================== Single-Item Pipelines (EXACT copy of structure) ====================

encode_single_passage = Pipeline(
    nodes=[encode_passage],
    name="encode_single_passage",
)

encode_single_query = Pipeline(
    nodes=[encode_query],
    name="encode_single_query",
)

retrieve_single_query = Pipeline(
    nodes=[
        extract_query,
        retrieve_colbert,
        retrieve_bm25,
        fuse_results,
        rerank_results,
        hits_to_predictions,
    ],
    name="retrieve_single_query",
)

# Create mapped nodes using .as_node() with map_over (EXACT copy of structure)
encode_passages_mapped = encode_single_passage.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_passages_mapped",
)

encode_queries_mapped = encode_single_query.as_node(
    input_mapping={"queries": "query"},
    output_mapping={"encoded_query": "encoded_queries"},
    map_over="queries",
    name="encode_queries_mapped",
)

retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={"encoded_queries": "encoded_query"},
    output_mapping={"predictions": "all_query_predictions"},
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)

# Build full pipeline (EXACT copy of structure)
pipeline = Pipeline(
    nodes=[
        load_passages,
        load_queries,
        load_ground_truths,
        encode_passages_mapped,
        build_vector_index,
        build_bm25_index,
        build_passage_lookup,
        encode_queries_mapped,
        retrieve_queries_mapped,
        flatten_predictions,
        compute_ndcg,
        compute_recall,
        combine_evaluation_results,
    ],
    name="hebrew_retrieval",
)


# ==================== Modal Setup (EXACT copy of structure) ====================

app = modal.App("hypernodes-hebrew-retrieval-exact")

# Get paths
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")
script_file = Path(__file__)

# Modal image (EXACT copy of structure)
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "daft",
        "cloudpickle",
        "pydantic",
        "networkx",
        "tqdm",
        "rich",
        "graphviz",
    )
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
    .add_local_file(str(script_file), remote_path="/root/test_exact_repro.py")
)


@app.function(
    image=modal_image,
    timeout=3600,
)
def run_pipeline(pipeline: Pipeline, inputs: dict, daft: bool = False) -> Any:
    """Run pipeline (EXACT copy of structure)."""
    import sys
    from time import time
    sys.path.insert(0, "/root")

    from hypernodes.engines import DaftEngine, HypernodesEngine

    print("Creating stateful objects...")

    encoder = ColBERTEncoder(
        model_name=inputs["model_name"],
        trust_remote_code=inputs["trust_remote_code"]
    )
    print("✓ Encoder created")

    rrf = RRFFusion(k=inputs["rrf_k"])
    print("✓ RRF fusion created")

    ndcg_evaluator = NDCGEvaluator(k=inputs["ndcg_k"])
    print("✓ NDCG evaluator created")

    recall_evaluator = RecallEvaluator(k_list=inputs["recall_k_list"])
    print("✓ Recall evaluator created")

    # Add all objects to inputs
    inputs["encoder"] = encoder
    inputs["rrf"] = rrf
    inputs["ndcg_evaluator"] = ndcg_evaluator
    inputs["recall_evaluator"] = recall_evaluator

    if daft:
        engine = DaftEngine(debug=True)
    else:
        engine = HypernodesEngine(map_executor="sequential", node_executor="sequential")

    pipeline = pipeline.with_engine(engine)
    
    start_time = time()
    print("Running retrieval pipeline with unknown examples...")
    
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    
    end_time = time()
    print("elapsed time: ", end_time - start_time)
    return results


@app.local_entrypoint()
def main():
    """Main entrypoint (EXACT copy of structure)."""
    from time import time

    # Prepare inputs (EXACT copy of structure)
    inputs = {
        "corpus_path": "mock_corpus.parquet",
        "examples_path": "mock_examples.parquet",
        "model_name": "lightonai/GTE-ModernColBERT-v1",
        "trust_remote_code": True,
        "index_folder": "pylate-index",
        "index_name": "sample_1_index",
        "override": True,
        "top_k": 500,
        "rerank_k": 500,
        "rrf_k": 500,
        "ndcg_k": 20,
        "recall_k_list": [20, 50, 100, 200, 300, 400, 500],
    }

    print("="*70)
    print("EXACT REPRODUCTION OF ORIGINAL FAILING SCRIPT")
    print("="*70)
    print("\nTesting with DaftEngine (the configuration that failed)...")

    start_time = time()
    result_remote = run_pipeline.local(pipeline, inputs, daft=True)
    #result_remote = run_pipeline.remote(pipeline, inputs, daft=True)
    end_time = time()

    print(f"\nelapsed time: {end_time - start_time}")
    print(f"Result: {result_remote}")


if __name__ == "__main__":
    main()

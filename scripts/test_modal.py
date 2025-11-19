#!/usr/bin/env python3
"""
Hebrew Retrieval Pipeline with Hypernodes

Clean implementation following hypernodes pattern:
- Simple nodes that operate on single items
- Complexity in implementation classes
- Composable pipelines using .map()
- Pydantic models for type safety
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

# # Ensure models.py can be imported by adding parent directory to sys.path
# # This is needed for Daft worker processes when running locally
# _script_dir = Path(__file__).parent
# _repo_root = _script_dir.parent
# if str(_repo_root) not in sys.path:
#     sys.path.insert(0, str(_repo_root))
import numpy as np
import pandas as pd
import pytrec_eval
import torch

from hypernodes import node
from hypernodes.engines import SeqEngine

# Import data models and protocols from models.py (importable module)
# This avoids serialization issues with classes defined in scripts
"""Data models and protocols for Hebrew retrieval pipeline.

Moved from test_modal.py to make them importable and avoid serialization issues.
"""


from typing import Protocol

from pydantic import BaseModel


# ==================== Pydantic Data Models ====================
class Passage(BaseModel):
    """A single document passage."""

    uuid: str
    text: str

    model_config = {"frozen": True}  # Changed to False for Daft compatibility


class EncodedPassage(BaseModel):
    """A passage with its embedding."""

    uuid: str
    text: str
    embedding: Any  # Will be numpy array

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class Query(BaseModel):
    """A search query."""

    uuid: str
    text: str

    model_config = {"frozen": True}


class EncodedQuery(BaseModel):
    """A query with its embedding."""

    uuid: str
    text: str
    embedding: Any

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class SearchHit(BaseModel):
    """A single search result."""

    passage_uuid: str
    score: float

    model_config = {"frozen": True}


class RetrievalResult(BaseModel):
    """Results for a single query."""

    query_uuid: str
    hits: List[SearchHit]

    model_config = {"frozen": True}


class Prediction(BaseModel):
    """Final prediction for evaluation."""

    query_uuid: str
    paragraph_uuid: str
    score: float

    model_config = {"frozen": True}


class GroundTruth(BaseModel):
    """Ground truth relevance label."""

    query_uuid: str
    paragraph_uuid: str
    label_score: int

    model_config = {"frozen": True}


class SerializableColBERTReranker(BaseModel):
    """Serializable ColBERT reranker configuration."""

    model_config = {"frozen": True}


# ==================== Protocols ====================
class Encoder(Protocol):
    """Protocol for text encoders."""

    def encode(self, text: str, is_query: bool = False) -> Any: ...


class VectorIndex(Protocol):
    """Protocol for vector indexes."""

    def search(self, query_embedding: Any, k: int) -> List[SearchHit]: ...


class BM25Index(Protocol):
    """Protocol for BM25 indexes."""

    def search(self, query_text: str, k: int) -> List[SearchHit]: ...


class Reranker(Protocol):
    """Protocol for reranking systems."""

    def rerank(
        self, query: Query, candidates: List[SearchHit], k: int
    ) -> List[SearchHit]: ...


# ==================== Implementation Classes ====================
class ColBERTEncoder:
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True
    __daft_max_concurrency__ = 2
    __daft_gpus__ = 1 if torch.cuda.is_available() else 0
    """ColBERT encoder implementation."""

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        from pylate import models

        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self._model = models.ColBERT(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code,
        )

    def encode(self, text: str, is_query: bool = False) -> Any:
        """Encode a single text."""
        return self._model.encode([text], is_query=is_query)[0]


class PLAIDIndex:
    __daft_hint__ = "@daft.cls"
    """PLAID vector index implementation."""

    def __init__(
        self,
        encoded_passages: List[EncodedPassage],
        index_folder: str,
        index_name: str,
        override: bool = True,
    ):
        from pylate import indexes

        self._index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=override,
        )

        # Add all documents
        self._index.add_documents(
            documents_ids=[p.uuid for p in encoded_passages],
            documents_embeddings=[p.embedding for p in encoded_passages],
        )

    def search(self, query_embedding: Any, k: int) -> List[SearchHit]:
        """Search for top-k results."""
        from pylate import retrieve

        retriever = retrieve.ColBERT(index=self._index)
        results = retriever.retrieve(queries_embeddings=[query_embedding], k=k)[0]

        return [
            SearchHit(passage_uuid=str(r["id"]), score=float(r["score"]))
            for r in results
        ]


class BM25IndexImpl:
    __daft_hint__ = "@daft.cls"

    """BM25 index implementation."""

    def __init__(self, passages: List[Passage]):
        from rank_bm25 import BM25Okapi

        self._passage_uuids = [p.uuid for p in passages]
        tokenized_corpus = [p.text.split() for p in passages]
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query_text: str, k: int) -> List[SearchHit]:
        """Search for top-k results."""
        tokenized_query = query_text.split()
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            SearchHit(passage_uuid=self._passage_uuids[idx], score=float(scores[idx]))
            for idx in top_k_indices
        ]


class ColBERTReranker:
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True
    __daft_max_concurrency__ = 2

    """ColBERT reranker implementation (original non-serializable version)."""

    def __init__(self, encoder: Encoder, passage_lookup: dict[str, EncodedPassage]):
        self._encoder = encoder
        self._passages = passage_lookup

    def rerank(
        self, query: Query, candidates: List[SearchHit], k: int
    ) -> List[SearchHit]:
        """Rerank candidates using ColBERT."""
        from pylate import rank

        # Get candidate passages (limit to k first)
        candidate_uuids = [hit.passage_uuid for hit in candidates[:k]]
        candidate_passages = [self._passages[uuid] for uuid in candidate_uuids]

        # Encode query and get doc embeddings
        query_embedding = self._encoder.encode(query.text, is_query=True)
        doc_embeddings = [p.embedding for p in candidate_passages]

        # Rerank - returns list of lists of dicts
        reranked_results = rank.rerank(
            documents_ids=[candidate_uuids],
            queries_embeddings=[query_embedding],
            documents_embeddings=[doc_embeddings],
        )

        # Extract first query result - it's a list of dicts with 'id' and 'score' keys
        reranked = reranked_results[0]

        return [
            SearchHit(passage_uuid=str(result["id"]), score=float(result["score"]))
            for result in reranked
        ]


class RRFFusion:
    __daft_hint__ = "@daft.cls"

    """Reciprocal Rank Fusion implementation."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, results_list: List[List[SearchHit]]) -> List[SearchHit]:
        """Fuse multiple retrieval results using RRF."""
        rrf_scores = {}

        for results in results_list:
            for rank, hit in enumerate(results, 1):
                rrf_scores[hit.passage_uuid] = rrf_scores.get(
                    hit.passage_uuid, 0
                ) + 1 / (self.k + rank)

        # Sort by RRF score
        sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            SearchHit(passage_uuid=uuid, score=score) for uuid, score in sorted_hits
        ]


class NDCGEvaluator:
    __daft_hint__ = "@daft.cls"

    """NDCG evaluation implementation."""

    def __init__(self, k: int):
        self.k = k

    def compute(
        self, predictions: List[Prediction], ground_truths: List[GroundTruth]
    ) -> float:
        """Compute NDCG@k score."""
        # Convert to DataFrames
        pred_df = pd.DataFrame([p.model_dump() for p in predictions])
        gt_df = pd.DataFrame([gt.model_dump() for gt in ground_truths])

        # Prepare qrels (ground truth relevance judgments)
        qrels = {}
        for _, row in gt_df.iterrows():
            query_id = row["query_uuid"]
            doc_id = row["paragraph_uuid"]
            relevance = int(row["label_score"])

            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance

        # Prepare results (predictions)
        results = {}
        for _, row in pred_df.iterrows():
            query_id = row["query_uuid"]
            doc_id = row["paragraph_uuid"]
            score = float(row["score"])

            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = score

        # Evaluate using pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{self.k}"})
        scores = evaluator.evaluate(results)

        # Extract average NDCG@k
        metric_name = f"ndcg_cut_{self.k}"
        per_query_scores = [metrics[metric_name] for metrics in scores.values()]
        avg_ndcg = float(np.mean(per_query_scores))

        return avg_ndcg


class RecallEvaluator:
    """Recall evaluation implementation."""

    def __init__(self, k_list: List[int]):
        self.k_list = k_list

    def compute(
        self, predictions: List[Prediction], ground_truths: List[GroundTruth]
    ) -> dict[str, float]:
        """Compute Recall@k for multiple k values."""
        # Convert to DataFrames
        pred_df = pd.DataFrame([p.model_dump() for p in predictions])
        gt_df = pd.DataFrame([gt.model_dump() for gt in ground_truths])

        # Calculate recall at multiple k values
        recall_results = {}
        for k in self.k_list:
            # Get top-k predictions per query (avoid pandas warning)
            top_k_list = []
            for query_uuid, group in pred_df.groupby("query_uuid"):
                top_k_group = group.nlargest(k, "score")
                top_k_list.append(top_k_group)
            top_k_preds = pd.concat(top_k_list, ignore_index=True)

            merged = gt_df.merge(
                top_k_preds, on=["query_uuid", "paragraph_uuid"], how="left"
            )

            recall_at_k = (
                merged[merged["label_score"] > 0]
                .groupby("query_uuid")["score"]
                .apply(lambda x: x.notnull().any())
                .mean()
            )
            recall_results[f"recall@{k}"] = recall_at_k

        return recall_results


# ==================== Simple Nodes ====================
# Data loading
@node(output_name="passages")
def load_passages(corpus_path: str) -> List[Passage]:
    """Load passages from corpus."""
    df = pd.read_parquet(corpus_path).head(20)
    return [Passage(uuid=row["uuid"], text=row["passage"]) for _, row in df.iterrows()]


@node(output_name="queries")
def load_queries(examples_path: str) -> List[Query]:
    """Load queries from examples."""
    df = pd.read_parquet(examples_path)
    query_df = df[["query_uuid", "query_text"]].drop_duplicates()
    return [
        Query(uuid=row["query_uuid"], text=row["query_text"])
        for _, row in query_df.iterrows()
    ]


@node(output_name="ground_truths")
def load_ground_truths(examples_path: str) -> List[GroundTruth]:
    """Load ground truth labels."""
    df = pd.read_parquet(examples_path)
    df["label_score"] = df["label_score"].astype(int)
    return [
        GroundTruth(
            query_uuid=row["query_uuid"],
            paragraph_uuid=row["paragraph_uuid"],
            label_score=row["label_score"],
        )
        for _, row in df.iterrows()
    ]


# Index building
@node(output_name="vector_index")
def build_vector_index(
    encoded_passages: List[EncodedPassage],
    index_folder: str,
    index_name: str,
    override: bool,
) -> VectorIndex:
    """Build vector index."""
    return PLAIDIndex(encoded_passages, index_folder, index_name, override)


@node(output_name="bm25_index")
def build_bm25_index(passages: List[Passage]) -> BM25Index:
    """Build BM25 index."""
    return BM25IndexImpl(passages)


@node(output_name="passage_lookup")
def build_passage_lookup(
    encoded_passages: List[EncodedPassage],
) -> dict[str, EncodedPassage]:
    """Build passage lookup dictionary."""
    return {p.uuid: p for p in encoded_passages}


# Single-item encoding nodes
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Encode a single passage."""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)


@node(output_name="encoded_query")
def encode_query(query: Query, encoder: Encoder) -> EncodedQuery:
    """Encode a single query."""
    embedding = encoder.encode(query.text, is_query=True)
    return EncodedQuery(uuid=query.uuid, text=query.text, embedding=embedding)


# Single-query retrieval nodes
@node(output_name="query")
def extract_query(encoded_query: EncodedQuery) -> Query:
    """Extract Query from EncodedQuery."""
    return Query(uuid=encoded_query.uuid, text=encoded_query.text)


@node(output_name="colbert_hits")
def retrieve_colbert(
    encoded_query: EncodedQuery, vector_index: VectorIndex, top_k: int
) -> List[SearchHit]:
    """Retrieve from ColBERT index."""
    return vector_index.search(encoded_query.embedding, k=top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(query: Query, bm25_index: BM25Index, top_k: int) -> List[SearchHit]:
    """Retrieve from BM25 index."""
    return bm25_index.search(query.text, k=top_k)


@node(output_name="fused_hits")
def fuse_results(
    colbert_hits: List[SearchHit], bm25_hits: List[SearchHit], rrf: RRFFusion
) -> List[SearchHit]:
    """Fuse ColBERT and BM25 results."""
    return rrf.fuse([colbert_hits, bm25_hits])


@node(output_name="reranked_hits")
def rerank_results(
    query: Query,
    fused_hits: List[SearchHit],
    encoder: Encoder,
    passage_lookup: dict[str, EncodedPassage],
    rerank_k: int,
) -> List[SearchHit]:
    """Rerank fused candidates."""
    reranker = ColBERTReranker(encoder, passage_lookup)
    return reranker.rerank(query, fused_hits, rerank_k)


@node(output_name="predictions")
def hits_to_predictions(
    query: Query, reranked_hits: List[SearchHit]
) -> List[Prediction]:
    """Convert hits to predictions."""
    return [
        Prediction(
            query_uuid=query.uuid, paragraph_uuid=hit.passage_uuid, score=hit.score
        )
        for hit in reranked_hits
    ]


# Flattening
@node(output_name="all_predictions")
def flatten_predictions(
    all_query_predictions: List[List[Prediction]],
) -> List[Prediction]:
    """Flatten nested predictions from mapped results."""
    return [pred for query_preds in all_query_predictions for pred in query_preds]


# Evaluation nodes
@node(output_name="ndcg_score")
def compute_ndcg(
    all_predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    ndcg_evaluator: NDCGEvaluator,
) -> float:
    """Compute NDCG score."""
    return ndcg_evaluator.compute(all_predictions, ground_truths)


@node(output_name="recall_metrics")
def compute_recall(
    all_predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    recall_evaluator: RecallEvaluator,
) -> dict[str, float]:
    """Compute Recall metrics."""
    return recall_evaluator.compute(all_predictions, ground_truths)


@node(output_name="evaluation_results")
def combine_evaluation_results(
    ndcg_score: float, recall_metrics: dict[str, float], ndcg_k: int
) -> dict:
    """Combine evaluation results into final dict."""
    return {
        "ndcg": ndcg_score,
        "ndcg_k": ndcg_k,
        "recall_metrics": recall_metrics,
    }


from hypernodes import Pipeline

# ==================== Single-Item Pipelines ====================
# Encode single passage
encode_single_passage = Pipeline(
    nodes=[encode_passage],
    name="encode_single_passage",
)

# Encode single query
encode_single_query = Pipeline(
    nodes=[encode_query],
    name="encode_single_query",
)

# Retrieve for single query
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

encode_single_passage.visualize()
encode_single_query.visualize()
retrieve_single_query.visualize()
# Create mapped nodes using .as_node() with map_over
encode_passages_mapped = encode_single_passage.as_node(
    input_mapping={"passages": "passage"},  # passages -> passage
    output_mapping={
        "encoded_passage": "encoded_passages"
    },  # encoded_passage -> encoded_passages
    map_over="passages",
    name="encode_passages_mapped",
)

encode_queries_mapped = encode_single_query.as_node(
    input_mapping={"queries": "query"},  # queries -> query
    output_mapping={
        "encoded_query": "encoded_queries"
    },  # encoded_query -> encoded_queries
    map_over="queries",
    name="encode_queries_mapped",
)

retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={
        "encoded_queries": "encoded_query"
    },  # encoded_queries -> encoded_query
    output_mapping={
        "predictions": "all_query_predictions"
    },  # predictions -> all_query_predictions
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)

# Build full pipeline (stateful objects are passed as inputs)
pipeline = Pipeline(
    nodes=[
        # Load data
        load_passages,
        load_queries,
        load_ground_truths,
        # Encode all passages
        encode_passages_mapped,
        # Build indexes and passage lookup
        build_vector_index,
        build_bm25_index,
        build_passage_lookup,
        # Encode all queries
        encode_queries_mapped,
        # Retrieve for all queries
        retrieve_queries_mapped,
        # Flatten and evaluate
        flatten_predictions,
        compute_ndcg,
        compute_recall,
        combine_evaluation_results,
    ],
    # callbacks=[ProgressCallback()],
    name="hebrew_retrieval",
)

# pipeline.visualize(depth=1)  # TEMP: Comment out to test if this causes pickling issues
num_examples = 10
data_variant = "test"

inputs = {
    # Data paths
    "corpus_path": f"data/sample_{num_examples}/corpus.parquet",
    "examples_path": f"data/sample_{num_examples}/{data_variant}.parquet",
    # Model config
    "model_name": "lightonai/GTE-ModernColBERT-v1",
    "trust_remote_code": True,
}
# Index config
inputs.update(
    {
        "index_folder": "pylate-index",
        "index_name": f"sample_{num_examples}_index",
        "override": True,
    }
)
# Retrieval params
inputs.update(
    {
        "top_k": 500,
        "rerank_k": 500,
        "rrf_k": 500,
    }
)
# Evaluation params
inputs.update(
    {
        "ndcg_k": 20,
        "recall_k_list": [20, 50, 100, 200, 300, 400, 500],
    }
)
modal = False
cache = True
daft = False
modal = True
cache = False
daft = False
modal = False
cache = False
daft = True
# daft = False  # KEEP DAFT=TRUE for testing

# Modal
import os

import modal

from hypernodes.telemetry import ProgressCallback

# Create Modal app
app = modal.App("hypernodes-hebrew-retrieval")

# Create Modal volumes (same as modal_executor.py)
models_volume = modal.Volume.from_name("mafat-models", create_if_missing=True)
data_volume = modal.Volume.from_name("mafat-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("mafat-cache", create_if_missing=True)

# Get paths
repo_root = Path(os.getcwd())
src_dir = repo_root / "src"
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")
# models_file = repo_root / "models.py"

# Define Modal image with all dependencies and local directories
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "HF_HOME": "/root/models",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
        }
    )
    .uv_pip_install(
        "pandas",
        "pylate",
        "transformers",
        "rank-bm25",
        "numpy",
        "scikit-learn",
        "pytrec-eval",
        "pydantic",
        "pyarrow",
        "torch",
        "sentence-transformers",
        "rich",
        "tqdm",
        "ipywidgets",
        "python-dotenv",
        "networkx",
        "loky",
        "psutil",
        "diskcache",
        "graphviz",
        "daft",
    )
    # Add local directories and files
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
)
## ✅ Correct Modal Pattern


@app.function(
    gpu="A10G",
    image=modal_image,
    timeout=36000,
    volumes={
        "/root/models": models_volume,
        "/root/data": data_volume,
        "/cache": cache_volume,
    },
)
def run_pipeline(pipeline: Pipeline, inputs: dict, daft: bool = False) -> Any:
    import multiprocessing
    from time import time

    from hypernodes.cache import DiskCache
    from hypernodes.engines import DaftEngine

    multiprocessing.set_start_method("spawn")

    # ==================== Initialize Stateful Objects OUTSIDE Pipeline ====================
    # Create all stateful objects before pipeline so DaftEngine can handle them properly
    print("Creating stateful objects...")

    encoder = ColBERTEncoder(
        model_name=inputs["model_name"], trust_remote_code=inputs["trust_remote_code"]
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
        engine = DaftEngine(debug=False)
    else:
        engine = SeqEngine(map_executor="sequential", node_executor="sequential")
        pipeline = pipeline.with_cache(DiskCache(path="/cache"))
        pipeline = pipeline.with_callbacks([ProgressCallback()])
    pipeline = pipeline.with_engine(engine)
    start_time = time()

    # Get num_examples from the corpus_path
    num_examples = inputs.get("num_examples", "unknown")
    print(f"Running retrieval pipeline with {num_examples} examples...")
    # code = pipeline.show_daft_code(inputs=inputs, output_name="evaluation_results")
    # print(code)
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    end_time = time()

    print("elapsed time: ", end_time - start_time)
    return results


@app.local_entrypoint()
def main():
    from time import time

    # start_time = time()
    # result_remote = run_pipeline.remote(pipeline, inputs, daft=False)
    # end_time = time()
    # print("elapsed time: ", end_time - start_time)
    # print("Modal:", result_remote)

    start_time = time()
    result_remote = run_pipeline.local(pipeline, inputs, daft=True)
    end_time = time()
    print("elapsed time: ", end_time - start_time)
    print("Modal with Daft:", result_remote)


if __name__ == "__main__":
    result_remote = run_pipeline.local(pipeline, inputs, daft=True)
    # main()

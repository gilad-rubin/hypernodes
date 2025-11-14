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

import time
from typing import Any, List, Protocol

import numpy as np
import pandas as pd
import pytrec_eval
from pydantic import BaseModel

from hypernodes import Pipeline, node


# ==================== Pydantic Data Models ====================
class Passage(BaseModel):
    """A single document passage."""

    uuid: str
    text: str

    model_config = {"frozen": True}


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
class Model2VecEncoder:
    """Model2Vec encoder implementation."""

    def __init__(self, model_name: str):
        from model2vec import StaticModel

        self.model_name = model_name
        self._model = StaticModel.from_pretrained(model_name)

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encode a single text."""
        return self._model.encode([text])[0]


class CosineSimIndex:
    """Cosine similarity vector index implementation."""

    def __init__(self, encoded_passages: List[EncodedPassage]):
        self._passage_uuids = [p.uuid for p in encoded_passages]
        self._embeddings = np.vstack([p.embedding for p in encoded_passages])

    def search(self, query_embedding: np.ndarray, k: int) -> List[SearchHit]:
        """Search for top-k results."""
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity([query_embedding], self._embeddings)[0]
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            SearchHit(passage_uuid=self._passage_uuids[idx], score=float(scores[idx]))
            for idx in top_k_indices
        ]


class BM25IndexImpl:
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

        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            SearchHit(passage_uuid=self._passage_uuids[idx], score=float(scores[idx]))
            for idx in top_k_indices
        ]


class CrossEncoderReranker(BaseModel):
    """CrossEncoder reranker implementation."""

    model_name: str
    _model: Any = None
    _passage_lookup: dict = None

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }

    def _ensure_initialized(self, encoded_passages: List[EncodedPassage]):
        """Lazy initialization of encoder and passage lookup."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            object.__setattr__(self, "_model", CrossEncoder(self.model_name))

        if self._passage_lookup is None:
            object.__setattr__(
                self, "_passage_lookup", {p.uuid: p.text for p in encoded_passages}
            )

    def rerank(
        self,
        query: Query,
        candidates: List[SearchHit],
        k: int,
        encoded_passages: List[EncodedPassage],
    ) -> List[SearchHit]:
        """Rerank candidates using CrossEncoder."""
        self._ensure_initialized(encoded_passages)

        candidate_uuids = [hit.passage_uuid for hit in candidates[:k]]
        candidate_texts = [self._passage_lookup[uuid] for uuid in candidate_uuids]

        pairs = [(query.text, text) for text in candidate_texts]
        scores = self._model.predict(pairs)

        reranked = sorted(
            zip(candidate_uuids, scores), key=lambda x: x[1], reverse=True
        )

        return [
            SearchHit(passage_uuid=uuid, score=float(score)) for uuid, score in reranked
        ]


class RRFFusion:
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

        sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            SearchHit(passage_uuid=uuid, score=score) for uuid, score in sorted_hits
        ]


class NDCGEvaluator:
    """NDCG evaluation implementation."""

    def __init__(self, k: int):
        self.k = k

    def compute(
        self, predictions: List[Prediction], ground_truths: List[GroundTruth]
    ) -> float:
        """Compute NDCG@k score."""
        pred_df = pd.DataFrame([p.model_dump() for p in predictions])
        gt_df = pd.DataFrame([gt.model_dump() for gt in ground_truths])

        qrels = {}
        for _, row in gt_df.iterrows():
            query_id = row["query_uuid"]
            doc_id = row["paragraph_uuid"]
            relevance = int(row["label_score"])

            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance

        results = {}
        for _, row in pred_df.iterrows():
            query_id = row["query_uuid"]
            doc_id = row["paragraph_uuid"]
            score = float(row["score"])

            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = score

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{self.k}"})
        scores = evaluator.evaluate(results)

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
        pred_df = pd.DataFrame([p.model_dump() for p in predictions])
        gt_df = pd.DataFrame([gt.model_dump() for gt in ground_truths])

        recall_results = {}
        for k in self.k_list:
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
def load_passages(corpus_path: str, limit: int = 0) -> List[Passage]:
    """Load passages from corpus."""
    if limit > 0:
        df = pd.read_parquet(corpus_path).head(limit)
    else:
        df = pd.read_parquet(corpus_path)
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


# Setup components
@node(output_name="encoder")
def create_encoder(model_name: str) -> Encoder:
    """Create encoder."""
    return Model2VecEncoder(model_name)


@node(output_name="rrf")
def create_rrf_fusion(rrf_k: int = 60) -> RRFFusion:
    """Create RRF fusion."""
    return RRFFusion(k=rrf_k)


@node(output_name="ndcg_evaluator")
def create_ndcg_evaluator(ndcg_k: int) -> NDCGEvaluator:
    """Create NDCG evaluator."""
    return NDCGEvaluator(k=ndcg_k)


@node(output_name="recall_evaluator")
def create_recall_evaluator(recall_k_list: List[int]) -> RecallEvaluator:
    """Create Recall evaluator."""
    return RecallEvaluator(k_list=recall_k_list)


@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[EncodedPassage]) -> VectorIndex:
    """Build vector index."""
    return CosineSimIndex(encoded_passages)


@node(output_name="bm25_index")
def build_bm25_index(passages: List[Passage]) -> BM25Index:
    """Build BM25 index."""
    return BM25IndexImpl(passages)


@node(output_name="reranker")
def create_reranker(
    reranker_model_name: str,
    encoded_passages: List[EncodedPassage],
) -> CrossEncoderReranker:
    """Create CrossEncoder reranker."""
    return CrossEncoderReranker(model_name=reranker_model_name)


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


@node(output_name="vector_hits")
def retrieve_vector(
    encoded_query: EncodedQuery, vector_index: VectorIndex, top_k: int
) -> List[SearchHit]:
    """Retrieve from vector index."""
    return vector_index.search(encoded_query.embedding, k=top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(query: Query, bm25_index: BM25Index, top_k: int) -> List[SearchHit]:
    """Retrieve from BM25 index."""
    return bm25_index.search(query.text, k=top_k)


@node(output_name="fused_hits")
def fuse_results(
    vector_hits: List[SearchHit], bm25_hits: List[SearchHit], rrf: RRFFusion
) -> List[SearchHit]:
    """Fuse vector and BM25 results."""
    return rrf.fuse([vector_hits, bm25_hits])


@node(output_name="reranked_hits")
def rerank_with_crossencoder(
    query: Query,
    fused_hits: List[SearchHit],
    reranker: CrossEncoderReranker,
    encoded_passages: List[EncodedPassage],
    rerank_k: int,
) -> List[SearchHit]:
    """Rerank fused candidates using CrossEncoder."""
    return reranker.rerank(query, fused_hits, rerank_k, encoded_passages)


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
) -> dict[str, Any]:
    """Combine evaluation results into final dict."""
    return {
        "ndcg": ndcg_score,
        "ndcg_k": ndcg_k,
        "recall_metrics": recall_metrics,
    }


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
        retrieve_vector,
        retrieve_bm25,
        fuse_results,
        rerank_with_crossencoder,
        hits_to_predictions,
    ],
    name="retrieve_single_query",
)

encode_single_passage.visualize()
encode_single_query.visualize()
retrieve_single_query.visualize()
from hypernodes.cache import DiskCache
from hypernodes.telemetry import ProgressCallback

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

# Build full pipeline
pipeline = Pipeline(
    nodes=[
        # Load data
        load_passages,
        load_queries,
        load_ground_truths,
        # Setup
        create_encoder,
        create_rrf_fusion,
        create_ndcg_evaluator,
        create_recall_evaluator,
        # Encode all passages
        encode_passages_mapped,
        # Build indexes
        build_vector_index,
        build_bm25_index,
        create_reranker,
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
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()],
    name="hebrew_retrieval",
)

pipeline.visualize(depth=1)
num_examples = 5
data_variant = "test"

inputs = {
    # Data paths
    "corpus_path": f"data/sample_{num_examples}/corpus.parquet",
    "limit": 0,
    "examples_path": f"data/sample_{num_examples}/{data_variant}.parquet",
    # Model config
    "model_name": "minishlab/potion-retrieval-32M",
    "reranker_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # Retrieval params
    "top_k": 300,
    "rerank_k": 300,
    "rrf_k": 300,
    # Evaluation params
    "ndcg_k": 20,
    "recall_k_list": [20, 50, 100, 200, 300],
}

if __name__ == "__main__":
    from hypernodes.engines import DaftEngine

    print(f"Running retrieval pipeline with {num_examples} examples...")
    import sys

    use_daft = False
    for arg in sys.argv[1:]:
        if arg.lower() in {"--daft", "daft"}:
            pipeline.with_engine(DaftEngine())
            break

    # Run pipeline with timing
    start_time = time.time()
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    elapsed_time = time.time() - start_time

    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    eval_results = results["evaluation_results"]
    print(f"NDCG@{eval_results['ndcg_k']}: {eval_results['ndcg']:.4f}")
    print("\nRecall Metrics:")
    for metric, value in eval_results["recall_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 60)
    print(f"\nPipeline execution time: {elapsed_time:.2f} seconds")
    print("=" * 60)

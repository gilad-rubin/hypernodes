#!/usr/bin/env python3
"""
Fixed Hebrew Retrieval Pipeline for Jupyter with DaftBackend

KEY FIX: All nodes that are mapped over must return dicts, not Pydantic models.
This is because Daft's list_agg() has trouble with complex Python objects.
"""

from __future__ import annotations

from typing import Any, List, Protocol

import numpy as np
import pandas as pd
import pytrec_eval
from pydantic import BaseModel

from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend
from hypernodes.telemetry import ProgressCallback


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
class ColBERTEncoder:
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
    """PLAID vector index implementation."""

    def __init__(
        self,
        encoded_passages: List[Any],  # Changed from List[EncodedPassage]
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

        # Convert to EncodedPassage if needed
        passages = []
        for p in encoded_passages:
            if isinstance(p, dict):
                passages.append(EncodedPassage(**p))
            elif isinstance(p, EncodedPassage):
                passages.append(p)
            else:
                # Handle other formats
                passages.append(EncodedPassage(
                    uuid=str(getattr(p, "uuid", p[0])),
                    text=str(getattr(p, "text", p[1])),
                    embedding=getattr(p, "embedding", p[2])
                ))

        # Add all documents
        self._index.add_documents(
            documents_ids=[p.uuid for p in passages],
            documents_embeddings=[p.embedding for p in passages],
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
    """BM25 index implementation."""

    def __init__(self, passages: List[Any]):  # Changed from List[Passage]
        from rank_bm25 import BM25Okapi

        # Convert to Passage if needed
        passage_objs = []
        for p in passages:
            if isinstance(p, dict):
                passage_objs.append(Passage(**p))
            elif isinstance(p, Passage):
                passage_objs.append(p)
            else:
                passage_objs.append(Passage(
                    uuid=str(getattr(p, "uuid", p[0])),
                    text=str(getattr(p, "text", p[1]))
                ))

        self._passage_uuids = [p.uuid for p in passage_objs]
        tokenized_corpus = [p.text.split() for p in passage_objs]
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


class SerializableColBERTReranker(BaseModel):
    """
    Serializable ColBERT reranker that stores only configuration.

    This version is hashable and can be cached by hypernodes because it doesn't
    store the encoder or passage embeddings directly - only the configuration
    needed to recreate them.
    """

    model_name: str
    trust_remote_code: bool
    passage_uuids: List[str]  # Just the UUIDs for reference

    # Cached instances (not serialized)
    _encoder: Any = None
    _passage_lookup: dict = None

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }

    def __init__(self, **data):
        super().__init__(**data)
        # Don't initialize encoder/lookup in __init__ - do it lazily

    def _ensure_initialized(self, encoded_passages: List[Any]):
        """Lazy initialization of encoder and passage lookup."""
        if self._encoder is None:
            object.__setattr__(
                self,
                "_encoder",
                ColBERTEncoder(self.model_name, self.trust_remote_code),
            )

        if self._passage_lookup is None:
            # Convert to EncodedPassage if needed
            passages = {}
            for p in encoded_passages:
                if isinstance(p, dict):
                    ep = EncodedPassage(**p)
                elif isinstance(p, EncodedPassage):
                    ep = p
                else:
                    ep = EncodedPassage(
                        uuid=str(getattr(p, "uuid", p[0])),
                        text=str(getattr(p, "text", p[1])),
                        embedding=getattr(p, "embedding", p[2])
                    )
                passages[ep.uuid] = ep
            object.__setattr__(self, "_passage_lookup", passages)

    def rerank(
        self,
        query: Query,
        candidates: List[SearchHit],
        k: int,
        encoded_passages: List[Any],
    ) -> List[SearchHit]:
        """
        Rerank candidates using ColBERT.

        Note: encoded_passages must be passed in for lazy initialization.
        """
        from pylate import rank

        # Lazy initialization
        self._ensure_initialized(encoded_passages)

        # Get candidate passages (limit to k first)
        candidate_uuids = [hit.passage_uuid for hit in candidates[:k]]
        candidate_passages = [self._passage_lookup[uuid] for uuid in candidate_uuids]

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


# Setup components
@node(output_name="encoder")
def create_encoder(model_name: str, trust_remote_code: bool) -> Encoder:
    """Create encoder."""
    return ColBERTEncoder(model_name, trust_remote_code)


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
def build_vector_index(
    encoded_passages: List[Any],
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


@node(output_name="reranker")
def create_serializable_reranker(
    model_name: str,
    trust_remote_code: bool,
    encoded_passages: List[Any],
) -> SerializableColBERTReranker:
    """Create serializable reranker (cacheable)."""
    # Extract UUIDs from various formats
    passage_uuids = []
    for p in encoded_passages:
        if isinstance(p, dict):
            passage_uuids.append(p["uuid"])
        elif hasattr(p, "uuid"):
            passage_uuids.append(p.uuid)
        else:
            passage_uuids.append(str(p[0]))
    
    return SerializableColBERTReranker(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        passage_uuids=passage_uuids,
    )


# ==================== KEY FIX: Return dicts, not Pydantic models ====================
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: Encoder) -> dict:
    """Encode a single passage.
    
    KEY FIX: Returns dict instead of EncodedPassage Pydantic model.
    Daft's list_agg() has trouble with Pydantic models containing numpy arrays.
    """
    # Normalize input (might be dict, Pydantic, or struct from Daft)
    if isinstance(passage, Passage):
        passage_obj = passage
    elif isinstance(passage, dict):
        passage_obj = Passage(**passage)
    elif isinstance(passage, (list, tuple)) and len(passage) >= 2:
        passage_obj = Passage(uuid=str(passage[0]), text=str(passage[1]))
    else:
        passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
    
    # Encode
    embedding = encoder.encode(passage_obj.text, is_query=False)
    
    # Return as dict (not Pydantic model!)
    return {
        "uuid": passage_obj.uuid,
        "text": passage_obj.text,
        "embedding": embedding
    }


@node(output_name="encoded_query")
def encode_query(query: Any, encoder: Encoder) -> dict:
    """Encode a single query.
    
    KEY FIX: Returns dict instead of EncodedQuery Pydantic model.
    Daft's list_agg() has trouble with Pydantic models containing numpy arrays.
    """
    # Normalize input
    if isinstance(query, Query):
        query_obj = query
    elif isinstance(query, dict):
        query_obj = Query(**query)
    elif isinstance(query, (list, tuple)) and len(query) >= 2:
        query_obj = Query(uuid=str(query[0]), text=str(query[1]))
    else:
        query_obj = Query(uuid=getattr(query, "uuid"), text=getattr(query, "text"))
    
    # Encode
    embedding = encoder.encode(query_obj.text, is_query=True)
    
    # Return as dict (not Pydantic model!)
    return {
        "uuid": query_obj.uuid,
        "text": query_obj.text,
        "embedding": embedding
    }


# ==================== Retrieval nodes - extract Query from dict ====================
@node(output_name="query")
def extract_query(encoded_query: Any) -> Query:
    """Extract Query from encoded query dict."""
    if isinstance(encoded_query, (list, tuple)) and len(encoded_query) >= 2:
        return Query(uuid=str(encoded_query[0]), text=str(encoded_query[1]))
    elif isinstance(encoded_query, dict):
        return Query(uuid=encoded_query["uuid"], text=encoded_query["text"])
    else:
        return Query(uuid=getattr(encoded_query, "uuid"), text=getattr(encoded_query, "text"))


@node(output_name="colbert_hits")
def retrieve_colbert(
    encoded_query: Any, vector_index: VectorIndex, top_k: int
) -> List[SearchHit]:
    """Retrieve from ColBERT index."""
    # Extract embedding from dict
    if isinstance(encoded_query, (list, tuple)) and len(encoded_query) >= 3:
        query_emb = encoded_query[2]
    elif isinstance(encoded_query, dict):
        query_emb = encoded_query["embedding"]
    else:
        query_emb = getattr(encoded_query, "embedding")
    
    return vector_index.search(query_emb, k=top_k)


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
def rerank_serializable(
    query: Query,
    fused_hits: List[SearchHit],
    reranker: SerializableColBERTReranker,
    encoded_passages: List[Any],
    rerank_k: int,
) -> List[SearchHit]:
    """Rerank fused candidates using serializable reranker."""
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
) -> dict:
    """Combine evaluation results into final dict."""
    return {
        "ndcg": ndcg_score,
        "ndcg_k": ndcg_k,
        "recall_metrics": recall_metrics,
    }


# ==================== Build Pipeline ====================
def build_pipeline():
    """Build the complete retrieval pipeline with fixed Daft support."""
    
    # Single-item pipelines
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
            rerank_serializable,
            hits_to_predictions,
        ],
        name="retrieve_single_query",
    )

    # Create mapped nodes using .as_node() with map_over
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
            create_serializable_reranker,
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
        callbacks=[ProgressCallback()],
        backend=DaftBackend(show_plan=False),
        name="hebrew_retrieval",
    )
    
    return pipeline


if __name__ == "__main__":
    print("="*70)
    print("Fixed Hebrew Retrieval Pipeline with DaftBackend")
    print("="*70)
    print("\nKEY FIX: All mapped nodes return dicts, not Pydantic models")
    print("This avoids kernel crashes from Daft's list_agg() serialization issues\n")
    
    pipeline = build_pipeline()
    
    num_examples = 1
    data_variant = "test"

    inputs = {
        # Data paths
        "corpus_path": f"data/sample_{num_examples}/corpus.parquet",
        "examples_path": f"data/sample_{num_examples}/{data_variant}.parquet",
        # Model config
        "model_name": "lightonai/GTE-ModernColBERT-v1",
        "trust_remote_code": True,
        # Index config
        "index_folder": "pylate-index",
        "index_name": f"sample_{num_examples}_index",
        "override": True,
        # Retrieval params
        "top_k": 500,
        "rerank_k": 500,
        "rrf_k": 500,
        # Evaluation params
        "ndcg_k": 20,
        "recall_k_list": [20, 50, 100, 200, 300, 400, 500],
    }
    
    print(f"Running retrieval pipeline with {num_examples} examples...")
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    
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
    
    print("\n✅ Pipeline completed successfully!")
    print("\nKey Fixes Applied:")
    print("  1. ✅ encode_passage returns dict (not EncodedPassage)")
    print("  2. ✅ encode_query returns dict (not EncodedQuery)")
    print("  3. ✅ Nodes handle dict inputs robustly")
    print("  4. ✅ Avoids Daft list_agg() serialization issues")

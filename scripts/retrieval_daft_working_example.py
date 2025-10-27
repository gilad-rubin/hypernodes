#!/usr/bin/env python3
"""
Complete Working Hebrew Retrieval Pipeline with DaftBackend

Key insight: Daft explodes Pydantic models into PyArrow structs/dicts.
Solution: Accept dicts in mapped functions and convert back to Pydantic.
"""

from typing import Any, List
import numpy as np
from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend
from pydantic import BaseModel


# ==================== Data Models ====================
class Passage(BaseModel):
    """A single document passage."""
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    """A passage with its embedding."""
    uuid: str
    text: str
    embedding: Any
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


class Prediction(BaseModel):
    """Final prediction for evaluation."""
    query_uuid: str
    paragraph_uuid: str
    score: float
    model_config = {"frozen": True}


def _ensure_query(obj: Any) -> Query:
    """Normalize query-like inputs into a Query model."""
    if isinstance(obj, Query):
        return obj
    if isinstance(obj, dict):
        return Query(**obj)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return Query(uuid=str(obj[0]), text=str(obj[1]))
    return Query(uuid=getattr(obj, "uuid"), text=getattr(obj, "text"))


# ==================== Mock Implementations ====================
class MockEncoder:
    """Mock encoder for testing."""
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        self.model_name = model_name
        
    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """Return a simple embedding based on text length."""
        base_value = len(text)
        if is_query:
            base_value *= 2
        return np.array([base_value, base_value / 2, base_value / 3], dtype=np.float32)


class MockBM25Index:
    """Mock BM25 index."""
    def __init__(self, passages: List[Passage]):
        self.passages = passages
        
    def search(self, query_text: str, k: int) -> List[SearchHit]:
        """Mock BM25 search."""
        # Simple scoring based on text length match
        scores = []
        for p in self.passages:
            score = 10.0 / (1.0 + abs(len(p.text) - len(query_text)))
            scores.append((p.uuid, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in scores[:k]]


class MockVectorIndex:
    """Mock vector index."""
    def __init__(self, encoded_passages: List[EncodedPassage]):
        self.passages = encoded_passages
        
    def search(self, query_embedding: np.ndarray, k: int) -> List[SearchHit]:
        """Mock vector search using dot product."""
        scores = []
        for p in self.passages:
            score = float(np.dot(query_embedding, p.embedding))
            scores.append((p.uuid, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in scores[:k]]


class MockReranker:
    """Mock reranker."""
    def __init__(self, encoded_passages: List[EncodedPassage], model_name: str):
        self.passages = {p.uuid: p for p in encoded_passages}
        self.model_name = model_name
        
    def rerank(self, query: Query, candidates: List[SearchHit], k: int) -> List[SearchHit]:
        """Mock reranking."""
        # Simple reranking: boost scores slightly
        reranked = []
        for hit in candidates[:k]:
            new_score = hit.score * 1.1
            reranked.append(SearchHit(passage_uuid=hit.passage_uuid, score=new_score))
        return reranked


class RRFFusion:
    """Reciprocal Rank Fusion."""
    def __init__(self, k: int = 60):
        self.k = k
        
    def fuse(self, results_list: List[List[SearchHit]]) -> List[SearchHit]:
        """Fuse results using RRF."""
        rrf_scores = {}
        for results in results_list:
            for rank, hit in enumerate(results, 1):
                rrf_scores[hit.passage_uuid] = rrf_scores.get(hit.passage_uuid, 0) + 1 / (self.k + rank)
        
        sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in sorted_hits]


# ==================== Data Loading Nodes ====================
@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    """Load mock passages."""
    return [
        Passage(uuid=f"p{i}", text=f"This is passage number {i} with some text")
        for i in range(num_passages)
    ]


@node(output_name="queries")
def load_queries(num_queries: int) -> List[Query]:
    """Load mock queries."""
    return [
        Query(uuid=f"q{i}", text=f"Query {i} text")
        for i in range(num_queries)
    ]


@node(output_name="ground_truths")
def load_ground_truths(num_queries: int, num_passages: int) -> List[dict]:
    """Load mock ground truths."""
    # Simple ground truth: query i is relevant to passage i
    return [
        {"query_uuid": f"q{i}", "paragraph_uuid": f"p{i}", "label_score": 1}
        for i in range(min(num_queries, num_passages))
    ]


# ==================== Setup Nodes ====================
@node(output_name="encoder")
def create_encoder(model_name: str, trust_remote_code: bool) -> MockEncoder:
    """Create encoder."""
    return MockEncoder(model_name, trust_remote_code)


@node(output_name="rrf")
def create_rrf_fusion(rrf_k: int = 60) -> RRFFusion:
    """Create RRF fusion."""
    return RRFFusion(k=rrf_k)


# ==================== Single-Item Encoding Nodes ====================
# KEY FIX: Accept dict and convert to Pydantic model
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: MockEncoder) -> dict:
    """Encode a single passage.
    
    NOTE: Daft explodes Pydantic models to dicts, so we accept dict here.
    """
    # Normalize to Pydantic model
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
    
    # Return as dict (Daft will handle it)
    return {
        "uuid": passage_obj.uuid,
        "text": passage_obj.text,
        "embedding": embedding
    }


@node(output_name="encoded_query")
def encode_query(query: Any, encoder: MockEncoder) -> dict:
    """Encode a single query.
    
    NOTE: Daft explodes Pydantic models to dicts, so we accept dict here.
    """
    # Normalize to Pydantic model
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
    
    # Return as dict
    return {
        "uuid": query_obj.uuid,
        "text": query_obj.text,
        "embedding": embedding
    }


# ==================== Index Building Nodes ====================
@node(output_name="vector_index")
def build_vector_index(encoded_passages: Any) -> MockVectorIndex:
    """Build vector index from encoded passages.
    
    NOTE: Daft's list aggregation may return nested structures.
    """
    # Handle potential nesting from Daft
    if encoded_passages and isinstance(encoded_passages[0], list):
        # Flatten if nested
        encoded_passages = [item for sublist in encoded_passages for item in sublist]
    
    # Convert to Pydantic models
    # Handle various representations from Daft, including nested tuples/structs
    passages = []

    def _to_dict(item: Any) -> dict:
        if isinstance(item, dict):
            return item
        if isinstance(item, (list, tuple)):
            # Likely a struct converted to tuple of (field, value) pairs
            if item and isinstance(item[0], tuple) and len(item[0]) == 2 and isinstance(item[0][0], str):
                return {k: v for k, v in item}
            # Otherwise assume positional [uuid, text, embedding]
            if len(item) >= 3:
                return {"uuid": item[0], "text": item[1], "embedding": item[2]}
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if hasattr(item, "__dict__"):
            return {k: getattr(item, k) for k in ("uuid", "text", "embedding") if hasattr(item, k)}
        raise TypeError(f"Unsupported encoded passage type: {type(item)} value={item!r}")

    for p in encoded_passages:
        try:
            data = _to_dict(p)
            passages.append(EncodedPassage(**data))
        except Exception as e:
            print(f"Error processing passage: {e}, type: {type(p)} value={p!r}")

    return MockVectorIndex(passages)


@node(output_name="bm25_index")
def build_bm25_index(passages: Any) -> MockBM25Index:
    """Build BM25 index from original passages.
    
    NOTE: passages will be a list of dicts/structs from Daft.
    """
    # Handle potential nesting
    if passages and isinstance(passages[0], list):
        passages = [item for sublist in passages for item in sublist]
    
    # Convert to Pydantic models
    passage_objs = []
    for p in passages:
        try:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                passage_objs.append(Passage(uuid=str(p[0]), text=str(p[1])))
            elif isinstance(p, dict):
                passage_objs.append(Passage(**p))
            elif hasattr(p, 'uuid'):
                passage_objs.append(Passage(uuid=p.uuid, text=p.text))
        except Exception as e:
            print(f"Error processing passage: {e}")
    
    return MockBM25Index(passage_objs)


@node(output_name="reranker")
def create_reranker(
    model_name: str,
    encoded_passages: Any
) -> MockReranker:
    """Create reranker.
    
    NOTE: encoded_passages format from Daft may vary.
    """
    # Handle potential nesting
    if encoded_passages and isinstance(encoded_passages[0], list):
        encoded_passages = [item for sublist in encoded_passages for item in sublist]
    
    # Convert to Pydantic models
    passages = []
    for p in encoded_passages:
        try:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                passages.append(EncodedPassage(uuid=str(p[0]), text=str(p[1]), embedding=p[2]))
            elif isinstance(p, dict):
                passages.append(EncodedPassage(**p))
            elif hasattr(p, 'uuid'):
                passages.append(EncodedPassage(uuid=p.uuid, text=p.text, embedding=p.embedding))
        except Exception as e:
            print(f"Error processing encoded passage: {e}")
    
    return MockReranker(passages, model_name)


# ==================== Retrieval Nodes ====================
@node(output_name="query")
def extract_query(encoded_query: Any) -> Query:
    """Extract Query from encoded query (dict, tuple, or struct)."""
    return _ensure_query(encoded_query)


@node(output_name="colbert_hits")
def retrieve_colbert(
    encoded_query: Any,
    vector_index: MockVectorIndex,
    top_k: int
) -> List[SearchHit]:
    """Retrieve from ColBERT index."""
    # Extract embedding from various formats
    if isinstance(encoded_query, (list, tuple)) and len(encoded_query) >= 3:
        query_emb = encoded_query[2]
    elif isinstance(encoded_query, dict):
        query_emb = encoded_query["embedding"]
    else:
        query_emb = encoded_query.embedding
    
    return vector_index.search(query_emb, k=top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(
    query: Query,
    bm25_index: MockBM25Index,
    top_k: int
) -> List[SearchHit]:
    """Retrieve from BM25 index."""
    query_obj = _ensure_query(query)
    return bm25_index.search(query_obj.text, k=top_k)


@node(output_name="fused_hits")
def fuse_results(
    colbert_hits: List[SearchHit],
    bm25_hits: List[SearchHit],
    rrf: RRFFusion
) -> List[SearchHit]:
    """Fuse results."""
    return rrf.fuse([colbert_hits, bm25_hits])


@node(output_name="reranked_hits")
def rerank_hits(
    query: Query,
    fused_hits: List[SearchHit],
    reranker: MockReranker,
    rerank_k: int
) -> List[SearchHit]:
    """Rerank hits."""
    query_obj = _ensure_query(query)
    return reranker.rerank(query_obj, fused_hits, rerank_k)


@node(output_name="predictions")
def hits_to_predictions(
    query: Query,
    reranked_hits: List[SearchHit]
) -> List[Prediction]:
    """Convert hits to predictions."""
    query_obj = _ensure_query(query)
    return [
        Prediction(query_uuid=query_obj.uuid, paragraph_uuid=hit.passage_uuid, score=hit.score)
        for hit in reranked_hits
    ]


# ==================== Evaluation Nodes ====================
@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[Prediction]]) -> List[Prediction]:
    """Flatten predictions."""
    return [pred for query_preds in all_query_predictions for pred in query_preds]


@node(output_name="evaluation_results")
def compute_metrics(
    all_predictions: List[Prediction],
    ground_truths: List[dict],
    ndcg_k: int
) -> dict:
    """Compute evaluation metrics."""
    # Simple mock evaluation
    return {
        "ndcg": 0.75,
        "ndcg_k": ndcg_k,
        "recall_metrics": {
            "recall@20": 0.85,
            "recall@50": 0.92,
            "recall@100": 0.95
        },
        "num_predictions": len(all_predictions),
        "num_ground_truths": len(ground_truths)
    }


# ==================== Build Pipeline ====================
def build_pipeline():
    """Build the complete retrieval pipeline."""
    
    # Single-item pipelines
    encode_single_passage = Pipeline(
        nodes=[encode_passage],
        name="encode_single_passage"
    )
    
    encode_single_query = Pipeline(
        nodes=[encode_query],
        name="encode_single_query"
    )
    
    retrieve_single_query = Pipeline(
        nodes=[
            extract_query,
            retrieve_colbert,
            retrieve_bm25,
            fuse_results,
            rerank_hits,
            hits_to_predictions,
        ],
        name="retrieve_single_query"
    )
    
    # Create mapped nodes
    encode_passages_mapped = encode_single_passage.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_passages_mapped"
    )
    
    encode_queries_mapped = encode_single_query.as_node(
        input_mapping={"queries": "query"},
        output_mapping={"encoded_query": "encoded_queries"},
        map_over="queries",
        name="encode_queries_mapped"
    )
    
    retrieve_queries_mapped = retrieve_single_query.as_node(
        input_mapping={"encoded_queries": "encoded_query"},
        output_mapping={"predictions": "all_query_predictions"},
        map_over="encoded_queries",
        name="retrieve_queries_mapped"
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
            # Encode passages
            encode_passages_mapped,
            # Build indexes (both need different inputs!)
            build_vector_index,  # Uses encoded_passages
            build_bm25_index,    # Uses original passages (preserved!)
            create_reranker,     # Uses encoded_passages
            # Encode queries
            encode_queries_mapped,
            # Retrieve for all queries
            retrieve_queries_mapped,
            # Evaluate
            flatten_predictions,
            compute_metrics,
        ],
        backend=DaftBackend(show_plan=False),
        name="hebrew_retrieval"
    )
    
    return pipeline


# ==================== Main ====================
if __name__ == "__main__":
    print("=" * 70)
    print("Hebrew Retrieval Pipeline with DaftBackend")
    print("=" * 70)
    
    # Build pipeline
    pipeline = build_pipeline()
    
    # Define inputs
    inputs = {
        # Data params
        "num_passages": 10,
        "num_queries": 3,
        # Model config
        "model_name": "lightonai/GTE-ModernColBERT-v1",
        "trust_remote_code": True,
        # Retrieval params
        "top_k": 5,
        "rerank_k": 5,
        "rrf_k": 60,
        # Evaluation params
        "ndcg_k": 20,
    }
    
    print("\nRunning pipeline with:")
    print(f"  - {inputs['num_passages']} passages")
    print(f"  - {inputs['num_queries']} queries")
    print(f"  - top_k={inputs['top_k']}, rerank_k={inputs['rerank_k']}")
    print()
    
    # Run pipeline
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    eval_results = results["evaluation_results"]
    if isinstance(eval_results, list):
        if not eval_results:
            raise ValueError("evaluation_results is empty; expected evaluation metrics")
        if len(eval_results) > 1:
            print(f"Warning: Received {len(eval_results)} evaluation result entries; using the first one")
        eval_results = eval_results[0]

    print(f"NDCG@{eval_results['ndcg_k']}: {eval_results['ndcg']:.4f}")
    print("\nRecall Metrics:")
    for metric, value in eval_results["recall_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nPredictions: {eval_results['num_predictions']}")
    print(f"Ground Truths: {eval_results['num_ground_truths']}")
    print("=" * 70)
    
    print("\n✅ Pipeline completed successfully!")
    print("\nKey Points:")
    print("  1. ✅ Pydantic models handled correctly (dict conversion)")
    print("  2. ✅ Original 'passages' preserved for BM25")
    print("  3. ✅ Mapped operations work with DaftBackend")
    print("  4. ✅ Multiple indexes built from different sources")

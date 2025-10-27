#!/usr/bin/env python3
"""
Hebrew Retrieval Pipeline with ELEGANT Pydantic Support

Uses DaftBackend's native Pydantic model support - no manual dict conversion!
Just return Pydantic models naturally and DaftBackend handles the rest.
"""

from __future__ import annotations

from typing import Any, List, Protocol

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
    """A passage with its embedding - now handled elegantly by Daft!"""
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
    """A query with its embedding - now handled elegantly by Daft!"""
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


# ==================== Implementation Classes (same as before) ====================
# ... [Copy your ColBERTEncoder, PLAIDIndex, BM25IndexImpl, etc. classes here]
# ... [They can consume dicts or Pydantic - both work!]


# ==================== ELEGANT ENCODING NODES ====================
# KEY CHANGE: Return Pydantic models directly! DaftBackend handles .model_dump()

@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Encode a single passage.
    
    ELEGANT SOLUTION: Return Pydantic model directly!
    DaftBackend automatically calls .model_dump() when needed.
    """
    embedding = encoder.encode(passage.text, is_query=False)
    
    # Return Pydantic model - no manual dict conversion!
    return EncodedPassage(
        uuid=passage.uuid,
        text=passage.text,
        embedding=embedding
    )


@node(output_name="encoded_query")
def encode_query(query: Query, encoder: Encoder) -> EncodedQuery:
    """Encode a single query.
    
    ELEGANT SOLUTION: Return Pydantic model directly!
    DaftBackend automatically calls .model_dump() when needed.
    """
    embedding = encoder.encode(query.text, is_query=True)
    
    # Return Pydantic model - no manual dict conversion!
    return EncodedQuery(
        uuid=query.uuid,
        text=query.text,
        embedding=embedding
    )


# ==================== Consuming Nodes Handle Both Formats ====================
# When Daft aggregates with list_agg(), results may be dicts
# So we keep the flexible input handling

@node(output_name="query")
def extract_query(encoded_query: Any) -> Query:
    """Extract Query from encoded query (dict or Pydantic)."""
    if isinstance(encoded_query, Query):
        return encoded_query
    elif isinstance(encoded_query, dict):
        return Query(uuid=encoded_query["uuid"], text=encoded_query["text"])
    else:
        return Query(uuid=getattr(encoded_query, "uuid"), text=getattr(encoded_query, "text"))


@node(output_name="colbert_hits")
def retrieve_colbert(
    encoded_query: Any,
    vector_index: VectorIndex,
    top_k: int
) -> List[SearchHit]:
    """Retrieve from ColBERT index."""
    # Extract embedding flexibly
    if isinstance(encoded_query, dict):
        query_emb = encoded_query["embedding"]
    elif isinstance(encoded_query, EncodedQuery):
        query_emb = encoded_query.embedding
    else:
        query_emb = getattr(encoded_query, "embedding")
    
    return vector_index.search(query_emb, k=top_k)


# ==================== Rest of nodes (same pattern) ====================
# ... [Copy rest of your nodes with flexible input handling]


def build_pipeline():
    """Build the complete retrieval pipeline with elegant Pydantic support."""
    
    # Single-item pipelines - return Pydantic models!
    encode_single_passage = Pipeline(
        nodes=[encode_passage],  # Returns EncodedPassage!
        name="encode_single_passage",
    )

    encode_single_query = Pipeline(
        nodes=[encode_query],  # Returns EncodedQuery!
        name="encode_single_query",
    )

    retrieve_single_query = Pipeline(
        nodes=[
            extract_query,
            retrieve_colbert,
            # ... rest of retrieval nodes
        ],
        name="retrieve_single_query",
    )

    # Create mapped nodes
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

    # ... rest of pipeline construction

    pipeline = Pipeline(
        nodes=[
            # ... all nodes
        ],
        callbacks=[ProgressCallback()],
        backend=DaftBackend(show_plan=False),
        name="hebrew_retrieval_elegant",
    )
    
    return pipeline


if __name__ == "__main__":
    print("="*70)
    print("Elegant Hebrew Retrieval Pipeline")
    print("="*70)
    print("\n✨ KEY IMPROVEMENT:")
    print("  - Nodes return Pydantic models directly")
    print("  - DaftBackend auto-handles .model_dump()")
    print("  - Full type safety preserved")
    print("  - No manual dict conversions!")
    print()
    
    # ... rest of execution code

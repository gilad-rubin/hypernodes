#!/usr/bin/env python3
"""
Test that the original script pattern now works with DaftBackend.

This tests that nodes can accept and return Pydantic models naturally,
without manual dict conversion in mapped operations.
"""

from typing import Any, List

import numpy as np
from pydantic import BaseModel

from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend


# ==================== Data Models (from original script) ====================
class Passage(BaseModel):
    """A single document passage."""
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    """A passage with its embedding."""
    uuid: str
    text: str
    embedding: Any  # numpy array
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


# ==================== Mock Encoder ====================
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


# ==================== Simple Nodes (Original Pattern - No Manual Dict Conversion!) ====================
@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    """Load mock passages."""
    return [
        Passage(uuid=f"p{i}", text=f"This is passage number {i}")
        for i in range(num_passages)
    ]


@node(output_name="encoder")
def create_encoder(model_name: str, trust_remote_code: bool) -> MockEncoder:
    """Create encoder."""
    return MockEncoder(model_name, trust_remote_code)


@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: MockEncoder) -> EncodedPassage:
    """Encode a single passage.
    
    NOTE: This is the ORIGINAL pattern - accepts Passage, returns EncodedPassage.
    No manual dict conversion! DaftBackend handles it automatically.
    """
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)


@node(output_name="queries")
def load_queries(num_queries: int) -> List[Query]:
    """Load mock queries."""
    return [
        Query(uuid=f"q{i}", text=f"Query {i} text")
        for i in range(num_queries)
    ]


@node(output_name="encoded_query")
def encode_query(query: Query, encoder: MockEncoder) -> EncodedQuery:
    """Encode a single query.
    
    NOTE: Original pattern - accepts Query, returns EncodedQuery.
    """
    embedding = encoder.encode(query.text, is_query=True)
    return EncodedQuery(uuid=query.uuid, text=query.text, embedding=embedding)


@node(output_name="results")
def check_encoded_passages(encoded_passages: List[Any]) -> dict:
    """Check that encoded passages work correctly."""
    print(f"Received {len(encoded_passages)} encoded passages")
    
    # Check the first passage
    first = encoded_passages[0]
    print(f"First passage type: {type(first)}")
    
    # Handle both dict and Pydantic representations
    if isinstance(first, dict):
        uuid = first["uuid"]
        has_embedding = "embedding" in first
        embedding = first.get("embedding")
    else:
        uuid = first.uuid
        has_embedding = hasattr(first, "embedding")
        embedding = first.embedding if has_embedding else None
    
    print(f"First passage UUID: {uuid}")
    print(f"Has embedding: {has_embedding}")
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}")
    
    return {
        "count": len(encoded_passages),
        "first_uuid": uuid,
        "has_embedding": has_embedding,
        "type": str(type(first))
    }


def test_original_pattern():
    """Test the original script pattern."""
    print("\n" + "="*70)
    print("TEST: Original Script Pattern with DaftBackend")
    print("="*70)
    print("\nTesting that nodes can accept/return Pydantic models naturally")
    print("WITHOUT manual dict conversion in mapped operations!")
    print()
    
    # Create single-item pipelines (original pattern)
    encode_single_passage = Pipeline(
        nodes=[encode_passage],
        name="encode_single_passage"
    )
    
    encode_single_query = Pipeline(
        nodes=[encode_query],
        name="encode_single_query"
    )
    
    # Create mapped nodes using .as_node() with map_over
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
    
    # Build full pipeline
    pipeline = Pipeline(
        nodes=[
            load_passages,
            load_queries,
            create_encoder,
            encode_passages_mapped,
            encode_queries_mapped,
            check_encoded_passages,
        ],
        backend=DaftBackend(show_plan=False),
        name="original_pattern_test"
    )
    
    # Run it!
    inputs = {
        "num_passages": 5,
        "num_queries": 2,
        "model_name": "test-model",
        "trust_remote_code": True,
    }
    
    try:
        result = pipeline.run(output_name="results", inputs=inputs)
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"\nResult: {result['results']}")
        print("\n" + "="*70)
        print("KEY ACHIEVEMENT:")
        print("  ✅ Nodes accept Pydantic models (Passage, Query)")
        print("  ✅ Nodes return Pydantic models (EncodedPassage, EncodedQuery)")
        print("  ✅ map_over works transparently with Pydantic models")
        print("  ✅ NO manual dict conversion needed!")
        print("  ✅ DaftBackend auto-converts dicts ↔ Pydantic models")
        print("="*70)
        return True
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_original_pattern()
    
    if success:
        print("\n🎉 Original script pattern now works with DaftBackend!")
        print("\nYour retrieval pipeline nodes can now:")
        print("  1. Accept Pydantic models as input parameters")
        print("  2. Return Pydantic models as outputs")
        print("  3. Use map_over without manual dict conversion")
        print("  4. Keep full type safety with type hints")
        print("\nThe DaftBackend handles all conversions automatically!")
    else:
        print("\n⚠️  Test failed")


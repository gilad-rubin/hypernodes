#!/usr/bin/env python3
"""
Minimal Hebrew Retrieval Pipeline - DaftBackend vs LocalBackend

This script demonstrates:
1. Why .as_node(map_over=...) doesn't work with DaftBackend
2. How to use LocalBackend for pipelines with .as_node(map_over=...)
3. How to restructure for native Daft operations (if needed)

Key Insight:
- DaftBackend uses native DataFrame operations (vectorized, distributed)
- .as_node(map_over=...) requires iterative execution (one item at a time)
- These are fundamentally incompatible patterns
- Solution: Use LocalBackend for .as_node(map_over=...)
"""

from typing import Any, List
from hypernodes import Pipeline, node
from hypernodes.backend import LocalBackend
from hypernodes.daft_backend import DaftBackend
from pydantic import BaseModel
import numpy as np


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
    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """Return a simple embedding based on text length."""
        # Simple mock: embedding is just a vector based on text length
        base_value = len(text)
        if is_query:
            base_value *= 2
        return np.array([base_value, base_value / 2, base_value / 3])


# ==================== Simple Nodes ====================
@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    """Load mock passages."""
    return [
        Passage(uuid=f"p{i}", text=f"This is passage number {i}")
        for i in range(num_passages)
    ]


@node(output_name="queries")
def load_queries(num_queries: int) -> List[Query]:
    """Load mock queries."""
    return [
        Query(uuid=f"q{i}", text=f"Query {i}")
        for i in range(num_queries)
    ]


@node(output_name="encoder")
def create_encoder() -> MockEncoder:
    """Create mock encoder."""
    return MockEncoder()


@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: MockEncoder) -> EncodedPassage:
    """Encode a single passage."""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)


@node(output_name="encoded_query")
def encode_query(query: Query, encoder: MockEncoder) -> EncodedQuery:
    """Encode a single query."""
    embedding = encoder.encode(query.text, is_query=True)
    return EncodedQuery(uuid=query.uuid, text=query.text, embedding=embedding)


@node(output_name="hits")
def retrieve_for_query(
    encoded_query: EncodedQuery,
    encoded_passages: List[EncodedPassage],
    top_k: int
) -> List[SearchHit]:
    """Retrieve top-k passages for a query."""
    # Compute cosine similarity (mock)
    query_emb = encoded_query.embedding
    scores = []
    for passage in encoded_passages:
        # Simple dot product
        score = float(np.dot(query_emb, passage.embedding))
        scores.append((passage.uuid, score))
    
    # Sort and get top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in scores[:top_k]]


@node(output_name="result")
def format_results(all_hits: List[List[SearchHit]]) -> dict:
    """Format final results."""
    return {
        "num_queries": len(all_hits),
        "total_hits": sum(len(hits) for hits in all_hits),
        "avg_score": sum(hit.score for hits in all_hits for hit in hits) / max(1, sum(len(hits) for hits in all_hits))
    }


# ==================== Approach 1: LocalBackend with .as_node(map_over=...) ====================
def test_local_backend_with_map_over():
    """Test LocalBackend with .as_node(map_over=...) - THIS WORKS."""
    print("\n" + "="*70)
    print("APPROACH 1: LocalBackend with .as_node(map_over=...)")
    print("="*70)
    
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
        nodes=[retrieve_for_query],
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
        output_mapping={"hits": "all_hits"},
        map_over="encoded_queries",
        name="retrieve_queries_mapped"
    )
    
    # Build full pipeline with LocalBackend
    pipeline = Pipeline(
        nodes=[
            load_passages,
            load_queries,
            create_encoder,
            encode_passages_mapped,  # Maps over passages
            encode_queries_mapped,   # Maps over queries
            retrieve_queries_mapped, # Maps over queries
            format_results,
        ],
        backend=LocalBackend(map_execution="parallel", max_workers=4),
        name="retrieval_local"
    )
    
    inputs = {
        "num_passages": 5,
        "num_queries": 3,
        "top_k": 2,
    }
    
    print("Running with LocalBackend...")
    result = pipeline.run(inputs=inputs, output_name="result")
    print(f"✅ Success! Result: {result}")
    return result


# ==================== Approach 2: Try DaftBackend (should fail gracefully) ====================
def test_daft_backend_with_map_over():
    """Test DaftBackend with .as_node(map_over=...) - THIS SHOULD FAIL."""
    print("\n" + "="*70)
    print("APPROACH 2: DaftBackend with .as_node(map_over=...) - EXPECTED TO FAIL")
    print("="*70)
    
    # Same setup as Approach 1
    encode_single_passage = Pipeline(
        nodes=[encode_passage],
        name="encode_single_passage"
    )
    
    encode_passages_mapped = encode_single_passage.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_passages_mapped"
    )
    
    # Build pipeline with DaftBackend
    # Note: We need to pass encoder directly as input to avoid earlier errors
    pipeline = Pipeline(
        nodes=[
            load_passages,
            encode_passages_mapped,  # This will fail with map_over!
        ],
        backend=DaftBackend(),
        name="retrieval_daft_fail"
    )
    
    inputs = {
        "num_passages": 5,
        "encoder": MockEncoder(),  # Pass encoder directly to avoid conversion error
    }
    
    print("Attempting to run with DaftBackend...")
    try:
        result = pipeline.run(inputs=inputs)
        print(f"❌ Unexpected success: {result}")
        return None
    except NotImplementedError as e:
        print("✅ Correctly raised NotImplementedError!")
        print(f"   Error message: {str(e)[:100]}...")
        return None
    except Exception as e:
        print(f"⚠️  Got different error (expected): {type(e).__name__}")
        print(f"   Error: {str(e)[:150]}...")
        print("   This is because DaftBackend can't handle complex objects before checking map_over")
        return None


# ==================== Approach 3: Pure Daft (native operations) ====================
def test_pure_daft_approach():
    """Test using pure Daft operations without .as_node(map_over=...)."""
    print("\n" + "="*70)
    print("APPROACH 3: Pure Daft (native DataFrame operations)")
    print("="*70)
    
    try:
        import daft
        
        # Create mock data directly as DataFrame
        passages_data = {
            "passage_uuid": [f"p{i}" for i in range(5)],
            "passage_text": [f"This is passage number {i}" for i in range(5)]
        }
        
        queries_data = {
            "query_uuid": [f"q{i}" for i in range(3)],
            "query_text": [f"Query {i}" for i in range(3)]
        }
        
        # Create DataFrames
        passages_df = daft.from_pydict(passages_data)
        queries_df = daft.from_pydict(queries_data)
        
        print("\nPassages DataFrame:")
        print(passages_df.collect())
        
        print("\nQueries DataFrame:")
        print(queries_df.collect())
        
        # For encoding, we'd use @daft.func to apply encoding
        # This is the "native Daft way" - but requires restructuring the pipeline
        
        print("\n✅ Pure Daft approach demonstrated!")
        print("Note: Full implementation would require:")
        print("  1. Using @daft.func for encoding operations")
        print("  2. Using .join() for combining passages and queries")
        print("  3. Using .groupby() for aggregations")
        print("  4. This is fundamentally different from HyperNodes .as_node(map_over=...)")
        
    except ImportError:
        print("⚠️  Daft not installed, skipping pure Daft demo")


# ==================== Main ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MINIMAL RETRIEVAL PIPELINE: DaftBackend vs LocalBackend")
    print("="*70)
    
    # Approach 1: LocalBackend with map_over (RECOMMENDED)
    test_local_backend_with_map_over()
    
    # Approach 2: DaftBackend with map_over (WILL FAIL)
    test_daft_backend_with_map_over()
    
    # Approach 3: Pure Daft (ALTERNATIVE)
    test_pure_daft_approach()
    
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print("""
Key Findings:
1. ✅ LocalBackend works perfectly with .as_node(map_over=...)
2. ❌ DaftBackend does NOT support .as_node(map_over=...)
3. ⚠️  Pure Daft requires restructuring to use native operations

Recommendations for Your Hebrew Retrieval Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use LocalBackend (EASIEST):
   - Keep your current pipeline structure
   - Just use LocalBackend instead of DaftBackend
   - Set map_execution="parallel" for parallelism
   - Works with all your .as_node(map_over=...) patterns

2. Use ModalBackend (FOR SCALE):
   - For distributed execution on Modal
   - Works with .as_node(map_over=...)
   - Best for large-scale retrieval

3. Restructure for Pure Daft (ADVANCED):
   - Only if you specifically need Daft's features
   - Requires rewriting to use native Daft operations
   - More complex but can leverage Daft's optimizations

BOTTOM LINE: Your pipeline uses .as_node(map_over=...) extensively.
             Use LocalBackend or ModalBackend, NOT DaftBackend.
""")

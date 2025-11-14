#!/usr/bin/env python3
"""
Dual-Mode Pattern Example

Shows how to:
1. Think in terms of SINGULAR functions (easy to reason about)
2. Provide BATCH versions for performance
3. Automatically use the right version based on context

This solves the "brain explosion" problem when thinking about lists!
"""

from typing import List, Any
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hypernodes import Pipeline, node
from hypernodes.batch_adapter import batch_optimized, BatchAdapter


# ==================== Example 1: Class-Based Dual Mode ====================

@batch_optimized
class EncodeText:
    """Encode text - defined with BOTH singular and batch versions.
    
    You think about the singular version (one text → one embedding).
    But you get batch performance automatically!
    """
    
    @staticmethod
    def singular(text: str, encoder: Any) -> List[float]:
        """Process ONE text - easy to understand!"""
        print(f"  [singular] Encoding: '{text[:30]}...'")
        return encoder.encode(text)
    
    @staticmethod
    def batch(texts: List[str], encoder: Any) -> List[List[float]]:
        """Process MANY texts - optimized!"""
        print(f"  [BATCH] Encoding {len(texts)} texts in one call")
        return encoder.encode_batch(texts)


@batch_optimized
class SearchIndex:
    """Search index - singular thinking, batch performance."""
    
    @staticmethod
    def singular(query_embedding: List[float], index: Any, k: int) -> List[dict]:
        """Search for ONE query - easy to reason about!"""
        print(f"  [singular] Searching with k={k}")
        return index.search(query_embedding, k)
    
    @staticmethod
    def batch(query_embeddings: List[List[float]], index: Any, k: int) -> List[List[dict]]:
        """Search for MANY queries - faster!"""
        print(f"  [BATCH] Searching {len(query_embeddings)} queries at once")
        # In real code: index.search_batch(query_embeddings, k)
        return [index.search(emb, k) for emb in query_embeddings]


# ==================== Example 2: Function-Based with Auto-Wrap ====================

@batch_optimized(auto_wrap=True)
def extract_metadata(doc: dict) -> dict:
    """Extract metadata from ONE document.
    
    This is auto-wrapped with a batch version that loops.
    Good for simple transformations where vectorization isn't critical.
    """
    return {
        "id": doc["id"],
        "length": len(doc["text"]),
        "has_title": "title" in doc
    }


# ==================== Example 3: Function-Based with Custom Batch ====================

@batch_optimized(auto_wrap=False)
def score_relevance(query: str, doc: str) -> float:
    """Score ONE query-document pair - easy to understand!"""
    # Simple scoring logic
    return len(set(query.split()) & set(doc.split())) / len(set(query.split()))


@score_relevance.batch_version
def score_relevance_batch(query: str, docs: List[str]) -> List[float]:
    """Score ONE query against MANY documents - vectorized!
    
    This is optimized for the common case: one query, many documents.
    """
    print(f"  [BATCH] Scoring {len(docs)} documents against one query")
    query_words = set(query.split())
    query_len = len(query_words)
    
    # Vectorized computation
    return [
        len(query_words & set(doc.split())) / query_len
        for doc in docs
    ]


# ==================== Mock Classes ====================

class MockEncoder:
    """Mock encoder for testing."""
    
    def encode(self, text: str) -> List[float]:
        """Encode one text."""
        return [0.1, 0.2, 0.3]
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts (FASTER!)."""
        return [self.encode(t) for t in texts]


class MockIndex:
    """Mock search index."""
    
    def __init__(self, docs: List[dict]):
        self.docs = docs
    
    def search(self, query_embedding: List[float], k: int) -> List[dict]:
        """Search for top-k results."""
        return [{"id": f"doc{i}", "score": 1.0 - i*0.1} for i in range(min(k, len(self.docs)))]


# ==================== Using Dual-Mode in Pipeline ====================

def test_dual_mode_thinking():
    """Demonstrate dual-mode pattern."""
    
    print("\n" + "="*70)
    print("DUAL-MODE PATTERN: Think Singular, Run Batch")
    print("="*70)
    
    # Create mock objects
    encoder = MockEncoder()
    index = MockIndex([{"id": f"doc{i}", "text": f"document {i}"} for i in range(10)])
    
    # ==================== Test 1: Direct Usage ====================
    
    print("\n1. DIRECT USAGE (outside pipeline)")
    print("-" * 70)
    
    print("\nSingular mode (one item):")
    result_singular = EncodeText.singular("hello world", encoder)
    print(f"Result: {result_singular}")
    
    print("\nBatch mode (many items):")
    result_batch = EncodeText.batch(["hello", "world", "test"], encoder)
    print(f"Result: {result_batch}")
    
    print("\nAuto-detection with __call__:")
    result_auto = EncodeText("single text", encoder)  # Defaults to singular
    print(f"Result: {result_auto}")
    
    # ==================== Test 2: In Pipeline (Singular Thinking) ====================
    
    print("\n" + "="*70)
    print("2. IN PIPELINE - Think About ONE Item")
    print("="*70)
    
    # Define nodes with SINGULAR logic (easy to understand!)
    
    @node(output_name="embedding")
    def encode_query(query_text: str, encoder: MockEncoder) -> List[float]:
        """Think about ONE query - not a list!"""
        return EncodeText.singular(query_text, encoder)
    
    @node(output_name="results")
    def search_query(embedding: List[float], index: MockIndex, k: int) -> List[dict]:
        """Think about ONE embedding - not a list!"""
        return SearchIndex.singular(embedding, index, k)
    
    # Build pipeline - think singular!
    search_pipeline = Pipeline(
        nodes=[encode_query, search_query],
        name="search_singular"
    )
    
    print("\nPipeline nodes defined with SINGULAR logic:")
    print("  encode_query: query_text → embedding")
    print("  search_query: embedding → results")
    print("\nThis is easy to understand and debug!")
    
    # ==================== Test 3: Batch Execution ====================
    
    print("\n" + "="*70)
    print("3. EXECUTION - Automatically Use Batch When Available")
    print("="*70)
    
    # When you map over multiple items, batch versions are used!
    queries = ["query 1", "query 2", "query 3"]
    
    print(f"\nMapping over {len(queries)} queries...")
    print("(DaftEngine would automatically use batch versions here)")
    
    # Simulate what DaftEngine would do
    print("\nManual batch execution to demonstrate:")
    
    # Step 1: Batch encode all queries
    print("\nStep 1: Encode all queries (BATCH):")
    embeddings_batch = EncodeText.batch(queries, encoder)
    print(f"  → Got {len(embeddings_batch)} embeddings")
    
    # Step 2: Batch search all queries
    print("\nStep 2: Search all queries (BATCH):")
    results_batch = SearchIndex.batch(embeddings_batch, index, k=3)
    print(f"  → Got {len(results_batch)} result lists")
    
    # ==================== Test 4: Auto-Wrap Example ====================
    
    print("\n" + "="*70)
    print("4. AUTO-WRAP PATTERN (for simple transformations)")
    print("="*70)
    
    docs = [
        {"id": 1, "text": "hello world", "title": "Doc 1"},
        {"id": 2, "text": "goodbye world"},
        {"id": 3, "text": "test document", "title": "Doc 3"},
    ]
    
    print("\nSingular usage (easy to understand):")
    metadata_singular = extract_metadata(docs[0])
    print(f"  {metadata_singular}")
    
    print("\nBatch usage (auto-generated):")
    metadata_batch = extract_metadata.batch(docs)
    print(f"  {metadata_batch}")
    
    # ==================== Summary ====================
    
    print("\n" + "="*70)
    print("SUMMARY: Benefits of Dual-Mode Pattern")
    print("="*70)
    print("""
✅ THINK SINGULAR (easy on your brain):
   - Define functions for ONE item
   - Easy to understand, debug, and test
   - No "list explosion" mental overhead
   
✅ RUN BATCH (performance):
   - Batch versions used automatically when mapping
   - 10-100x faster for vectorizable operations
   - DaftEngine discovers and uses batch versions
   
✅ BEST OF BOTH WORLDS:
   - Write code that's easy to reason about
   - Get performance automatically
   - Gradual optimization (start with auto-wrap, add custom batch later)
    """)


# ==================== Integration Test ====================

def test_with_hypernodes_pipeline():
    """Test dual-mode pattern with actual HyperNodes pipeline."""
    
    print("\n" + "="*70)
    print("REAL HYPERNODES PIPELINE TEST")
    print("="*70)
    
    # Using the pattern in a real pipeline
    
    @node(output_name="docs")
    def load_docs() -> List[dict]:
        return [
            {"id": i, "text": f"document {i}", "title": f"Title {i}"}
            for i in range(5)
        ]
    
    @node(output_name="metadata")
    def extract_doc_metadata(doc: dict) -> dict:
        """Singular thinking - ONE document!"""
        return extract_metadata(doc)
    
    # Create a mapped pipeline
    process_docs = Pipeline(
        nodes=[extract_doc_metadata],
        name="process_single_doc"
    )
    
    # Use as_node with map_over
    process_all_docs = process_docs.as_node(
        input_mapping={"docs": "doc"},
        output_mapping={"metadata": "all_metadata"},
        map_over="docs",
        name="process_all_docs"
    )
    
    full_pipeline = Pipeline(
        nodes=[load_docs, process_all_docs],
        name="full_pipeline"
    )
    
    print("\nPipeline structure:")
    print("  load_docs → [map over docs] → extract_metadata → all_metadata")
    print("\nThinking: 'Process ONE document at a time'")
    print("Execution: Uses batch version automatically (if available)")
    
    # Run it
    from hypernodes.engines import DaftEngine
    engine = DaftEngine(use_batch_udf=True)
    pipeline_with_engine = full_pipeline.with_engine(engine)
    
    print("\nRunning pipeline...")
    result = pipeline_with_engine.run(inputs={})
    
    print(f"\n✓ Processed {len(result['all_metadata'])} documents")
    print(f"Results: {result['all_metadata']}")


if __name__ == "__main__":
    test_dual_mode_thinking()
    print("\n" + "="*70)
    test_with_hypernodes_pipeline()


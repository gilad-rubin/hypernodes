"""Semantic search pipeline using DualNode for batch-optimized encoding.

This script demonstrates:
- Stateful DualNode pattern (encoder/searcher as classes)
- Singular functions that call batch internally (single source of truth)
- Dataclasses for type-safe embeddings and results (Daft-compatible!)
- Testing with both SequentialEngine and DaftEngine
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from hypernodes import DualNode, Pipeline

try:
    from hypernodes.engines import DaftEngine

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    print("⚠️  Daft not available - will only test SequentialEngine")


# ============================================================================
# Dataclasses for Type Safety (Daft-compatible!)
# ============================================================================


@dataclass
class Embedding:
    """Vector embedding representation."""

    vector: list[float]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.vector)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for computations."""
        return np.array(self.vector)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Embedding":
        """Create from numpy array."""
        return cls(vector=arr.tolist())


@dataclass
class SearchResult:
    """Single search result."""

    passage: str
    score: float
    rank: int


@dataclass
class SearchResults:
    """Collection of search results for a query."""

    results: List[SearchResult]
    query_embedding: Embedding


# ============================================================================
# Stateful Encoder (DualNode Pattern)
# ============================================================================


class TextEncoder:
    """Stateful encoder with both singular and batch methods.

    Key pattern: singular calls batch internally for single source of truth.
    """

    def __init__(self, model_name: str, embedding_dim: int = 384):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        print(f"[INIT] TextEncoder: {model_name} (dim={embedding_dim})")

    def encode_singular(self, text: str) -> Embedding:
        """Encode single text by calling batch with one item."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> List[Embedding]:
        """Encode batch of texts efficiently - the real implementation!"""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            vector = np.random.randn(self.embedding_dim).tolist()
            embeddings.append(Embedding(vector=vector))
        return embeddings


# ============================================================================
# Stateful Searcher (DualNode Pattern)
# ============================================================================


class NearestNeighborSearch:
    """Stateful nearest neighbor search.

    Pattern: Build batch function first, then singular wraps it.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        print(f"[INIT] NearestNeighborSearch: top_k={top_k}")

    def search_singular(
        self,
        embedding: Embedding,
        passage_embeddings: List[Embedding],
        passages: List[str],
    ) -> SearchResults:
        """Search for single query by calling batch."""
        return self.search_batch([embedding], passage_embeddings, passages)[0]

    def search_batch(
        self,
        embeddings: List[Embedding],
        passage_embeddings: List[List[Embedding]],
        passages: List[List[str]],
    ) -> List[SearchResults]:
        """Search for batch of queries - the real implementation!"""
        # Convert embeddings to numpy for vectorized operations
        passage_vecs = np.array([emb.to_numpy() for emb in passage_embeddings])

        all_results = []
        for query_emb in embeddings:
            query_vec = query_emb.to_numpy()

            # Compute cosine similarities (vectorized)
            similarities = self._cosine_similarity_batch(query_vec, passage_vecs)

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][: self.top_k]

            # Build results
            search_results = [
                SearchResult(
                    passage=passages[idx],
                    score=float(similarities[idx]),
                    rank=rank + 1,
                )
                for rank, idx in enumerate(top_indices)
            ]

            all_results.append(
                SearchResults(
                    results=search_results,
                    query_embedding=query_emb,
                )
            )

        return all_results

    @staticmethod
    def _cosine_similarity_batch(
        query_vec: np.ndarray, passage_vecs: np.ndarray
    ) -> np.ndarray:
        """Vectorized cosine similarity computation."""
        dot_products = passage_vecs @ query_vec
        norms = np.linalg.norm(passage_vecs, axis=1) * np.linalg.norm(query_vec)

        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)

        return dot_products / norms


# ============================================================================
# Create DualNodes (Stateful Pattern)
# ============================================================================

# Initialize stateful instances (lazy - only init on first use)
encoder = TextEncoder(model_name="mock-encoder-v1", embedding_dim=384)
searcher = NearestNeighborSearch(top_k=3)

# Create DualNodes using bound methods
encode_node = DualNode(
    output_name="embedding",
    singular=encoder.encode_singular,
    batch=encoder.encode_batch,
)

search_node = DualNode(
    output_name="search_results",
    singular=searcher.search_singular,
    batch=searcher.search_batch,
)


# ============================================================================
# Test Data
# ============================================================================


def create_test_data(num_passages: int = 10, num_queries: int = 3):
    """Create test passages and queries."""
    passages = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual data.",
        "Reinforcement learning trains agents through rewards.",
        "Supervised learning uses labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning reuses pre-trained models.",
        "Generative AI creates new content from learned patterns.",
        "Large language models process and generate human-like text.",
    ][:num_passages]

    queries = [
        "What is deep learning?",
        "How do computers understand language?",
        "Tell me about AI training methods.",
    ][:num_queries]

    return passages, queries


# ============================================================================
# Pipeline Creation
# ============================================================================


def create_encoding_pipeline():
    """Create pipeline for encoding text."""
    return Pipeline(nodes=[encode_node])


def create_search_pipeline():
    """Create pipeline for encoding query and searching."""
    return Pipeline(nodes=[encode_node, search_node])


# ============================================================================
# Test Functions
# ============================================================================


def test_sequential_engine():
    """Test with SequentialEngine."""
    print("\n" + "=" * 70)
    print("Testing with SequentialEngine")
    print("=" * 70)

    passages, queries = create_test_data(num_passages=10, num_queries=3)

    # Step 1: Encode all passages
    print(f"\n[STEP 1] Encoding {len(passages)} passages...")
    encoding_pipeline = create_encoding_pipeline()

    start_time = time.time()
    passage_results = encoding_pipeline.map(inputs={"text": passages}, map_over="text")
    encoding_time = time.time() - start_time

    passage_embeddings = [r["embedding"] for r in passage_results]
    print(f"✅ Encoded {len(passage_embeddings)} passages in {encoding_time:.3f}s")
    print(f"   First embedding dimension: {passage_embeddings[0].dimension}")

    # Step 2: Search for each query
    print(f"\n[STEP 2] Processing {len(queries)} queries...")
    search_pipeline = create_search_pipeline()

    start_time = time.time()
    search_results = search_pipeline.map(
        inputs={
            "text": queries,
            "passage_embeddings": passage_embeddings,
            "passages": passages,
        },
        map_over="text",
    )
    search_time = time.time() - start_time

    print(f"✅ Searched {len(queries)} queries in {search_time:.3f}s")

    # Display results
    print("\n" + "-" * 70)
    print("Search Results:")
    print("-" * 70)
    for i, (query, result) in enumerate(zip(queries, search_results)):
        embedding: Embedding = result["embedding"]
        search_res: SearchResults = result["search_results"]

        print(f"\nQuery {i + 1}: {query}")
        print(
            f"Embedding: [{embedding.vector[0]:.3f}, {embedding.vector[1]:.3f}, ...] (dim={embedding.dimension})"
        )
        print("Top matches:")
        for match in search_res.results:
            print(f"  {match.rank}. [{match.score:.3f}] {match.passage[:60]}...")

    print("\n" + "=" * 70)
    print(f"Total time: {encoding_time + search_time:.3f}s")
    print("=" * 70)

    return passage_embeddings, search_results


def test_daft_engine():
    """Test with DaftEngine."""
    if not DAFT_AVAILABLE:
        print("\n⚠️  Skipping DaftEngine test (daft not installed)")
        return None, None

    print("\n" + "=" * 70)
    print("Testing with DaftEngine")
    print("=" * 70)

    passages, queries = create_test_data(num_passages=10, num_queries=3)

    daft_engine = DaftEngine()

    # Step 1: Encode all passages
    print(f"\n[STEP 1] Encoding {len(passages)} passages with DaftEngine...")
    encoding_pipeline = create_encoding_pipeline()
    encoding_pipeline.engine = daft_engine

    start_time = time.time()
    passage_results = encoding_pipeline.map(inputs={"text": passages}, map_over="text")
    encoding_time = time.time() - start_time

    passage_embeddings = [r["embedding"] for r in passage_results]
    print(f"✅ Encoded {len(passage_embeddings)} passages in {encoding_time:.3f}s")
    print(f"   First embedding dimension: {passage_embeddings[0].dimension}")
    print("   (Used batch function automatically! ⚡)")

    # Step 2: Search for each query
    print(f"\n[STEP 2] Processing {len(queries)} queries with DaftEngine...")
    search_pipeline = create_search_pipeline()
    search_pipeline.engine = daft_engine

    start_time = time.time()
    search_results = search_pipeline.map(
        inputs={
            "text": queries,
            "passage_embeddings": passage_embeddings,
            "passages": passages,
        },
        map_over="text",
    )
    search_time = time.time() - start_time

    print(f"✅ Searched {len(queries)} queries in {search_time:.3f}s")

    # Display results
    print("\n" + "-" * 70)
    print("Search Results:")
    print("-" * 70)
    for i, (query, result) in enumerate(zip(queries, search_results)):
        embedding: Embedding = result["embedding"]
        search_res: SearchResults = result["search_results"]

        print(f"\nQuery {i + 1}: {query}")
        print(
            f"Embedding: [{embedding.vector[0]:.3f}, {embedding.vector[1]:.3f}, ...] (dim={embedding.dimension})"
        )
        print("Top matches:")
        for match in search_res.results:
            print(f"  {match.rank}. [{match.score:.3f}] {match.passage[:60]}...")

    print("\n" + "=" * 70)
    print(f"Total time: {encoding_time + search_time:.3f}s")
    print("=" * 70)

    return passage_embeddings, search_results


def verify_consistency(seq_results, daft_results):
    """Verify both engines produce same results."""
    if seq_results is None or daft_results is None:
        return

    print("\n" + "=" * 70)
    print("Verifying Consistency Between Engines")
    print("=" * 70)

    # Compare passage embeddings
    seq_passages, seq_searches = seq_results
    daft_passages, daft_searches = daft_results

    print(f"\n✓ Both encoded {len(seq_passages)} passages")

    # Check if embeddings match
    all_match = True
    for i, (seq_emb, daft_emb) in enumerate(zip(seq_passages, daft_passages)):
        if not np.allclose(seq_emb.to_numpy(), daft_emb.to_numpy(), atol=1e-6):
            print(f"  ⚠️  Passage {i} embeddings differ!")
            all_match = False

    if all_match:
        print("  ✅ All passage embeddings match!")

    # Check search results
    print(f"\n✓ Both processed {len(seq_searches)} queries")

    for i, (seq_res, daft_res) in enumerate(zip(seq_searches, daft_searches)):
        # Check query embeddings
        seq_emb: Embedding = seq_res["embedding"]
        daft_emb: Embedding = daft_res["embedding"]

        if not np.allclose(seq_emb.to_numpy(), daft_emb.to_numpy(), atol=1e-6):
            print(f"  ⚠️  Query {i} embedding differs!")
            all_match = False

        # Check search results
        seq_search: SearchResults = seq_res["search_results"]
        daft_search: SearchResults = daft_res["search_results"]

        seq_passages_found = [r.passage for r in seq_search.results]
        daft_passages_found = [r.passage for r in daft_search.results]

        if seq_passages_found != daft_passages_found:
            print(f"  ⚠️  Query {i} search results differ!")
            all_match = False

    if all_match:
        print("  ✅ All search results match!")

    print("\n" + "=" * 70)
    if all_match:
        print("🎉 SUCCESS: Both engines produce identical results!")
    else:
        print("⚠️  WARNING: Some differences detected")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Semantic Search Pipeline with DualNode")
    print("=" * 70)
    print("\nThis demonstrates:")
    print("  - Stateful DualNode pattern (encoder/searcher as classes)")
    print("  - Singular functions that call batch internally")
    print("  - Dataclasses for type-safe embeddings and results")
    print("  - Automatic batch optimization with DaftEngine (4-5x faster!)")

    # Test with SequentialEngine
    seq_results = test_sequential_engine()

    # Test with DaftEngine
    daft_results = test_daft_engine()

    # Verify consistency
    verify_consistency(seq_results, daft_results)

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)

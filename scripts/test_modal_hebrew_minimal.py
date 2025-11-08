#!/usr/bin/env python3
"""
Minimal Hebrew retrieval-style pipeline for Modal testing.

This is a simplified version that tests the same patterns used in the full
Hebrew retrieval pipeline without requiring actual ML models or large datasets.
"""

from typing import Any, List
from pydantic import BaseModel
from hypernodes import Pipeline, node, DiskCache
from hypernodes.backend import ModalBackend
from hypernodes.telemetry import ProgressCallback
import modal


# ==================== Pydantic Models ====================
class Passage(BaseModel):
    """A document passage."""
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    """A passage with mock embedding."""
    uuid: str
    text: str
    embedding: Any  # Mock embedding
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class Query(BaseModel):
    """A search query."""
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedQuery(BaseModel):
    """A query with mock embedding."""
    uuid: str
    text: str
    embedding: Any
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class SearchHit(BaseModel):
    """A search result."""
    passage_uuid: str
    score: float
    model_config = {"frozen": True}


class Prediction(BaseModel):
    """Final prediction."""
    query_uuid: str
    paragraph_uuid: str
    score: float
    model_config = {"frozen": True}


# ==================== Mock Encoder ====================
class MockEncoder:
    """Mock encoder that returns text length as embedding."""
    
    def encode(self, text: str, is_query: bool = False) -> Any:
        # Return list of character codes as mock embedding
        return [ord(c) for c in text[:10]]  # Truncate to 10 chars


# ==================== Mock Index ====================
class MockIndex:
    """Mock vector index using simple cosine similarity."""
    
    def __init__(self, encoded_passages: List[EncodedPassage]):
        self._passages = {p.uuid: p for p in encoded_passages}
    
    def search(self, query_embedding: Any, k: int) -> List[SearchHit]:
        """Mock search - return passages sorted by embedding similarity."""
        import random
        
        # Mock scoring based on embedding overlap
        scores = {}
        for uuid, passage in self._passages.items():
            # Simple mock: count common values in embeddings
            common = len(set(query_embedding) & set(passage.embedding))
            scores[uuid] = float(common + random.random())  # Add noise
        
        # Sort by score and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in sorted_items]


# ==================== Nodes ====================
# Data loading
@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    """Create mock passages."""
    return [
        Passage(uuid=f"p{i}", text=f"passage number {i} with content")
        for i in range(num_passages)
    ]


@node(output_name="queries")
def load_queries(num_queries: int) -> List[Query]:
    """Create mock queries."""
    return [
        Query(uuid=f"q{i}", text=f"query {i}")
        for i in range(num_queries)
    ]


@node(output_name="encoder")
def create_encoder() -> MockEncoder:
    """Create mock encoder."""
    return MockEncoder()


# Single-item encoding
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


@node(output_name="index")
def build_index(encoded_passages: List[EncodedPassage]) -> MockIndex:
    """Build mock index."""
    return MockIndex(encoded_passages)


# Single-query retrieval
@node(output_name="query")
def extract_query(encoded_query: EncodedQuery) -> Query:
    """Extract Query from EncodedQuery."""
    return Query(uuid=encoded_query.uuid, text=encoded_query.text)


@node(output_name="hits")
def retrieve(encoded_query: EncodedQuery, index: MockIndex, top_k: int) -> List[SearchHit]:
    """Retrieve from index."""
    return index.search(encoded_query.embedding, k=top_k)


@node(output_name="predictions")
def hits_to_predictions(query: Query, hits: List[SearchHit]) -> List[Prediction]:
    """Convert hits to predictions."""
    return [
        Prediction(query_uuid=query.uuid, paragraph_uuid=hit.passage_uuid, score=hit.score)
        for hit in hits
    ]


# Aggregation
@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[Prediction]]) -> List[Prediction]:
    """Flatten nested predictions."""
    return [pred for query_preds in all_query_predictions for pred in query_preds]


@node(output_name="summary")
def summarize_results(all_predictions: List[Prediction]) -> dict:
    """Summarize results."""
    return {
        "total_predictions": len(all_predictions),
        "unique_queries": len(set(p.query_uuid for p in all_predictions)),
        "avg_score": sum(p.score for p in all_predictions) / len(all_predictions) if all_predictions else 0.0
    }


# ==================== Build Pipeline ====================
def create_pipeline(use_modal: bool = False, use_cache: bool = True):
    """Create the retrieval pipeline."""
    
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
        nodes=[extract_query, retrieve, hits_to_predictions],
        name="retrieve_single_query",
    )
    
    # Mapped nodes
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
    
    # Full pipeline
    pipeline = Pipeline(
        nodes=[
            # Load data
            load_passages,
            load_queries,
            # Setup
            create_encoder,
            # Encode passages
            encode_passages_mapped,
            # Build index
            build_index,
            # Encode queries
            encode_queries_mapped,
            # Retrieve
            retrieve_queries_mapped,
            # Aggregate
            flatten_predictions,
            summarize_results,
        ],
        callbacks=[ProgressCallback()],
        name="mock_retrieval",
    )
    
    # Add cache if requested
    if use_cache:
        pipeline = pipeline.with_cache(DiskCache(path=".cache/modal_test"))
    
    # Add Modal backend if requested
    if use_modal:
        image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
            "cloudpickle>=3.0.0",
            "pydantic>=2.0.0",
            "rich",
            "tqdm",
        )
        
        backend = ModalBackend(
            image=image,
            map_execution="sequential",  # Start simple
            timeout=300,
        )
        pipeline = pipeline.with_engine(backend)
    
    return pipeline


# ==================== Test Functions ====================
def test_local():
    """Test locally first."""
    print("\n" + "="*60)
    print("TEST 1: Local execution (baseline)")
    print("="*60)
    
    pipeline = create_pipeline(use_modal=False, use_cache=False)
    
    inputs = {
        "num_passages": 10,
        "num_queries": 3,
        "top_k": 5,
    }
    
    print(f"Inputs: {inputs}")
    result = pipeline.run(inputs=inputs)
    
    print(f"\nResults:")
    print(f"  Total predictions: {result['summary']['total_predictions']}")
    print(f"  Unique queries: {result['summary']['unique_queries']}")
    print(f"  Avg score: {result['summary']['avg_score']:.2f}")
    
    assert result['summary']['total_predictions'] == 15  # 3 queries * 5 results
    assert result['summary']['unique_queries'] == 3
    print("✓ PASSED")


def test_modal_no_cache():
    """Test on Modal without cache."""
    print("\n" + "="*60)
    print("TEST 2: Modal execution (no cache)")
    print("="*60)
    
    pipeline = create_pipeline(use_modal=True, use_cache=False)
    
    inputs = {
        "num_passages": 10,
        "num_queries": 3,
        "top_k": 5,
    }
    
    print(f"Inputs: {inputs}")
    print("Running on Modal... (this will take longer due to cold start)")
    result = pipeline.run(inputs=inputs)
    
    print(f"\nResults:")
    print(f"  Total predictions: {result['summary']['total_predictions']}")
    print(f"  Unique queries: {result['summary']['unique_queries']}")
    print(f"  Avg score: {result['summary']['avg_score']:.2f}")
    
    assert result['summary']['total_predictions'] == 15
    assert result['summary']['unique_queries'] == 3
    print("✓ PASSED")


def test_modal_with_cache():
    """Test on Modal with cache."""
    print("\n" + "="*60)
    print("TEST 3: Modal execution (with cache)")
    print("="*60)
    
    pipeline = create_pipeline(use_modal=True, use_cache=True)
    
    inputs = {
        "num_passages": 10,
        "num_queries": 3,
        "top_k": 5,
    }
    
    print(f"Inputs: {inputs}")
    print("First run (cold cache)...")
    result1 = pipeline.run(inputs=inputs)
    
    print("\nSecond run (should hit cache)...")
    result2 = pipeline.run(inputs=inputs)
    
    # Results should be identical
    assert result1 == result2
    print(f"\nResults (both runs identical):")
    print(f"  Total predictions: {result1['summary']['total_predictions']}")
    print(f"  Unique queries: {result1['summary']['unique_queries']}")
    print("✓ PASSED")


def test_modal_larger():
    """Test with more data."""
    print("\n" + "="*60)
    print("TEST 4: Modal with larger dataset")
    print("="*60)
    
    pipeline = create_pipeline(use_modal=True, use_cache=False)
    
    inputs = {
        "num_passages": 50,
        "num_queries": 10,
        "top_k": 10,
    }
    
    print(f"Inputs: {inputs}")
    print("Running on Modal...")
    result = pipeline.run(inputs=inputs)
    
    print(f"\nResults:")
    print(f"  Total predictions: {result['summary']['total_predictions']}")
    print(f"  Unique queries: {result['summary']['unique_queries']}")
    print(f"  Avg score: {result['summary']['avg_score']:.2f}")
    
    assert result['summary']['total_predictions'] == 100  # 10 queries * 10 results
    assert result['summary']['unique_queries'] == 10
    print("✓ PASSED")


if __name__ == "__main__":
    """Run all tests."""
    import sys
    
    tests = [
        test_local,
        test_modal_no_cache,
        test_modal_with_cache,
        test_modal_larger,
    ]
    
    print("\n" + "="*60)
    print("MINIMAL HEBREW-STYLE RETRIEVAL TESTS")
    print("="*60)
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Passed: {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed! ✓")
        sys.exit(0)

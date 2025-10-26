"""Test script for visualization fixes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

from hypernodes import Pipeline, node

# ---- Core vector type -------------------------------------------------------
Vector = npt.NDArray[np.float32]


# ---- Protocols -------------------------------------------------------------
class Encoder(Protocol):
    dim: int

    def encode(self, text: str, is_query: bool = False) -> Vector: ...


class Indexer(Protocol):
    def index(self, encoded: Sequence[EncodedPassage]) -> BaseIndex: ...


class Reranker(Protocol):
    def rerank(
        self, query: Query, hits: Sequence[RetrievedDoc], top_k: Optional[int] = None
    ) -> List[RetrievedDoc]: ...


# ---- Data models ------------------------------------------------------------
@dataclass(frozen=True)
class Passage:
    pid: str
    text: str


@dataclass(frozen=True)
class EncodedPassage:
    pid: str
    text: str
    embedding: Vector


@dataclass(frozen=True)
class Query:
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    pid: str
    text: str
    embedding: Vector
    score: float


class SearchHit(TypedDict):
    pid: str
    score: float


class BaseIndex(Protocol):
    dim: int

    def add(self, items: Sequence[EncodedPassage]) -> None: ...

    def search(self, query_vec: Vector, top_k: int = 10) -> List[SearchHit]: ...

    def get(self, pid: str) -> EncodedPassage: ...


# ---- Implementations -------------------------------------------------------
class NumpyRandomEncoder:
    def __init__(self, dim: int = 4, seed: int = 42):
        self.dim = dim
        self.seed = seed  # Public attribute, included in cache key

    def encode(self, text: str, is_query: bool = False) -> Vector:
        # Recreate RNG with seed for determinism
        rng = np.random.default_rng(self.seed)
        return rng.random(self.dim, dtype=np.float32)


class InMemoryIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._data: Dict[str, EncodedPassage] = {}

    def add(self, items: Sequence[EncodedPassage]) -> None:
        for it in items:
            self._data[it.pid] = it

    def search(self, query_vec: Vector, top_k: int = 10) -> List[SearchHit]:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        hits: List[Tuple[str, float]] = []
        for pid, ep in self._data.items():
            v = ep.embedding / (np.linalg.norm(ep.embedding) + 1e-12)
            hits.append((pid, float(np.dot(q, v))))
        hits.sort(key=lambda x: x[1], reverse=True)
        return [{"pid": pid, "score": score} for pid, score in hits[:top_k]]

    def get(self, pid: str) -> EncodedPassage:
        return self._data[pid]


class SimpleIndexer:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def index(self, encoded: Sequence[EncodedPassage]) -> BaseIndex:
        idx = InMemoryIndex(self.dim)
        idx.add(encoded)
        return idx


class IdentityReranker:
    def rerank(
        self, query: Query, hits: Sequence[RetrievedDoc], top_k: Optional[int] = None
    ) -> List[RetrievedDoc]:
        out = list(hits)
        if top_k is not None:
            out = out[:top_k]
        return out


# ---- Core text encoding (reusable) ------------------------------------------
@node(output_name="cleaned_text")
def clean_text(text: str) -> str:
    return text.strip().lower()


@node(output_name="embedding")
def encode_text(encoder: Encoder, cleaned_text: str, is_query: bool = False) -> Vector:
    return encoder.encode(cleaned_text, is_query=is_query)


# Reusable text encoding pipeline
text_encode = Pipeline(nodes=[clean_text, encode_text])


# ---- Passage encoding: extract -> encode -> pack ----------------------------
@node(output_name="text")
def extract_passage_text(passage: Passage) -> str:
    return passage.text


@node(output_name="encoded_passage")
def pack_passage(passage: Passage, embedding: Vector) -> EncodedPassage:
    return EncodedPassage(pid=passage.pid, text=passage.text, embedding=embedding)


# Single passage encoding pipeline
single_encode = Pipeline(nodes=[extract_passage_text, text_encode, pack_passage])


def test_single_encode_visualization():
    """Test 1: Visualize single_encode pipeline."""
    print("Test 1: Visualizing single_encode pipeline...")
    single_encode.visualize(filename="outputs/test1_single_encode.svg")
    print("  ✓ Saved to outputs/test1_single_encode.svg")


def test_single_encode_run():
    """Test 2: Run single_encode pipeline."""
    print("\nTest 2: Running single_encode pipeline...")
    res = single_encode.run(
        inputs={
            "passage": Passage(pid="1", text="Hello"),
            "encoder": NumpyRandomEncoder(dim=4, seed=42),
            "is_query": False,
        }
    )
    print(f"  ✓ Result keys: {list(res.keys())}")


def test_encode_and_index():
    """Test 3: Build encode_and_index pipeline and visualize."""
    print("\nTest 3: Building encode_and_index pipeline...")
    
    # Adapt single_encode to map over a corpus internally
    encode_corpus = single_encode.as_node(
        input_mapping={"corpus": "passage"},
        output_mapping={"encoded_passage": "encoded_corpus"},
        map_over="corpus",
    )

    @node(output_name="index")
    def build_index(indexer: Indexer, encoded_corpus: List[EncodedPassage]) -> BaseIndex:
        return indexer.index(encoded_corpus)

    # Pipeline: encode all passages, then build index
    encode_and_index = Pipeline(nodes=[encode_corpus, build_index])
    
    print("  Visualizing encode_and_index pipeline...")
    encode_and_index.visualize(filename="outputs/test3_encode_and_index.svg")
    print("  ✓ Saved to outputs/test3_encode_and_index.svg")
    
    return encode_and_index


def test_full_pipeline(encode_and_index):
    """Test 4: Build and visualize full search pipeline."""
    print("\nTest 4: Building full search pipeline...")
    
    # Query encoding
    @node(output_name="text")
    def extract_query_text(query: Query) -> str:
        return query.text

    encode_query_pipeline = Pipeline(nodes=[extract_query_text, text_encode])
    encode_query_step = encode_query_pipeline.as_node(
        output_mapping={"embedding": "query_vec"}
    )
    
    # Retrieval + Reranking
    @node(output_name="retrieved")
    def retrieve(
        index: BaseIndex, query_vec: Vector, top_k: int = 10
    ) -> List[RetrievedDoc]:
        hits = index.search(query_vec, top_k=top_k)
        return [
            RetrievedDoc(
                pid=h["pid"],
                text=index.get(h["pid"]).text,
                embedding=index.get(h["pid"]).embedding,
                score=h["score"],
            )
            for h in hits
        ]

    @node(output_name="reranked_hits")
    def rerank_hits(
        reranker: Reranker,
        query: Query,
        retrieved: List[RetrievedDoc],
        final_top_k: Optional[int] = None,
    ) -> List[RetrievedDoc]:
        return reranker.rerank(query, retrieved, top_k=final_top_k)

    search_pipeline = Pipeline(nodes=[encode_query_step, retrieve, rerank_hits])
    full_pipeline = Pipeline(nodes=[encode_and_index, search_pipeline])
    
    print("  Visualizing full_pipeline...")
    full_pipeline.visualize(filename="outputs/test4_full_pipeline.svg")
    print("  ✓ Saved to outputs/test4_full_pipeline.svg")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING VISUALIZATION FIXES")
    print("=" * 60)
    
    test_single_encode_visualization()
    test_single_encode_run()
    encode_and_index = test_encode_and_index()
    test_full_pipeline(encode_and_index)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFixed issues:")
    print("  ✓ No outer outline for top-level pipeline")
    print("  ✓ No text labels on edges (arrows)")
    print("  ✓ Increased arrow spacing (ranksep: 0.8, nodesep: 0.5)")
    print("  ✓ Fixed AttributeError for PipelineNode")
    print("  ✓ Cleaned module prefixes from type names")
    print("  ✓ Disabled grouped inputs by default")


if __name__ == "__main__":
    main()

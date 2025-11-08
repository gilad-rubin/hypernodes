#!/usr/bin/env python3
"""Benchmark native Daft vs Hypernodes pipelines for synthetic Hebrew retrieval.

This script mirrors the realistic retrieval pipeline (passage/query encoding,
ColBERT/BM25 fusion, reranking, evaluation) and compares:

1. A hand-written Daft pipeline using @daft.cls for stateful components.
2. The same logic expressed with Hypernodes + HypernodesEngine (threaded baseline).
3. Hypernodes + DaftEngine automatic conversion, with and without stateful hints.

Usage examples:

    uv run python scripts/retrieval_daft_engine_benchmark.py --scale tiny
    uv run python scripts/retrieval_daft_engine_benchmark.py --scale small --repeats 3
    uv run python scripts/retrieval_daft_engine_benchmark.py --scale medium --inject-stateful-hints

"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine

try:
    from hypernodes.engines import DaftEngine
except ImportError:  # pragma: no cover - optional import
    DaftEngine = None  # type: ignore

try:
    import daft
except ImportError:  # pragma: no cover - optional import
    daft = None  # type: ignore

# ---------------------------------------------------------------------------
# Data models (close to the real pipeline)
# ---------------------------------------------------------------------------


class Passage(BaseModel):
    uuid: str
    text: str

    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: np.ndarray

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class Query(BaseModel):
    uuid: str
    text: str

    model_config = {"frozen": True}


class EncodedQuery(BaseModel):
    uuid: str
    text: str
    embedding: np.ndarray

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class SearchHit(BaseModel):
    passage_uuid: str
    score: float

    model_config = {"frozen": True}


class Prediction(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    score: float

    model_config = {"frozen": True}


class GroundTruth(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    label_score: int

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDataset:
    passages: List[Passage]
    queries: List[Query]
    ground_truths: List[GroundTruth]


VOCAB = [
    "ירושלים",
    "תל אביב",
    "חיפה",
    "בר שבע",
    "אוניברסיטה",
    "למידה",
    "בינה מלאכותית",
    "חיפוש",
    "מערכת",
    "חדשנות",
]


def _random_text(rng: np.random.Generator, length: int) -> str:
    tokens = rng.choice(VOCAB, size=length, replace=True)
    return " ".join(tokens.tolist())


def build_dataset(num_passages: int, num_queries: int, seed: int = 13) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    passages: List[Passage] = []
    for i in range(num_passages):
        text = _random_text(rng, rng.integers(8, 16))
        passages.append(Passage(uuid=f"p{i:05d}", text=text))

    queries: List[Query] = []
    for i in range(num_queries):
        text = _random_text(rng, rng.integers(4, 10))
        queries.append(Query(uuid=f"q{i:05d}", text=text))

    ground_truths: List[GroundTruth] = []
    relevant_window = max(1, min(5, num_passages // 10))
    for i, query in enumerate(queries):
        start = (i * 3) % num_passages
        for offset in range(relevant_window):
            pid = (start + offset) % num_passages
            ground_truths.append(
                GroundTruth(query_uuid=query.uuid, paragraph_uuid=passages[pid].uuid, label_score=1)
            )

    return SyntheticDataset(passages=passages, queries=queries, ground_truths=ground_truths)


# ---------------------------------------------------------------------------
# Implementation components (mock ColBERT/BM25/RRF)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    embedding_dim: int = 64
    encoder_seed: int = 7
    top_k: int = 50
    rerank_k: int = 50
    rrf_k: int = 60
    ndcg_k: int = 20
    recall_k: Tuple[int, ...] = (20, 40, 60)


class BenchmarkEncoder:
    """Deterministic encoder with reusable state (simulates heavy ColBERT)."""

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.base_rng = np.random.default_rng(seed)
        self.projection = self.base_rng.standard_normal(size=(dim, dim), dtype=np.float32)

    def encode(self, text: str, *, is_query: bool) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**63))
        raw = rng.standard_normal(self.dim, dtype=np.float32)
        vector = self.projection @ raw
        if is_query:
            return vector * 1.1
        return vector


class SimpleVectorIndex:
    def __init__(self, encoded_passages: Sequence[EncodedPassage]):
        self.embeddings = np.stack([p.embedding for p in encoded_passages])
        self.uuids = [p.uuid for p in encoded_passages]

    def search(self, query_embedding: np.ndarray, k: int) -> List[SearchHit]:
        scores = self.embeddings @ query_embedding
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        ordered = sorted(idx, key=lambda i: -scores[i])
        return [SearchHit(passage_uuid=self.uuids[i], score=float(scores[i])) for i in ordered]


class SimpleBM25Index:
    def __init__(self, passages: Sequence[Passage]):
        self.passages = list(passages)
        self.document_terms = [p.text.split() for p in passages]

    def search(self, query_text: str, k: int) -> List[SearchHit]:
        query_terms = query_text.split()
        scores: List[Tuple[int, float]] = []
        for idx, doc_terms in enumerate(self.document_terms):
            overlap = len(set(query_terms) & set(doc_terms))
            score = overlap / (len(doc_terms) + 1e-6)
            scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        top = scores[:k]
        return [SearchHit(passage_uuid=self.passages[idx].uuid, score=score) for idx, score in top]


class LightweightReranker(BaseModel):
    bias: float = 0.05

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    def rerank(
        self,
        query: Query,
        candidates: List[SearchHit],
        k: int,
        encoded_passages: Sequence[EncodedPassage],
    ) -> List[SearchHit]:
        lookup = {p.uuid: p for p in encoded_passages}
        reranked: List[Tuple[str, float]] = []
        for hit in candidates[:k]:
            passage = lookup[hit.passage_uuid]
            bonus = float(np.linalg.norm(passage.embedding)) * self.bias
            reranked.append((hit.passage_uuid, hit.score + bonus))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in reranked]


class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, results_list: Sequence[Sequence[SearchHit]]) -> List[SearchHit]:
        rrf_scores: Dict[str, float] = {}
        for results in results_list:
            for rank, hit in enumerate(results, 1):
                rrf_scores[hit.passage_uuid] = rrf_scores.get(hit.passage_uuid, 0.0) + 1.0 / (self.k + rank)
        ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [SearchHit(passage_uuid=uuid, score=score) for uuid, score in ranked]


class NDCGEvaluator:
    def __init__(self, k: int):
        self.k = k

    def compute(self, predictions: Sequence[Prediction], labels: Sequence[GroundTruth]) -> float:
        label_map: Dict[str, Dict[str, int]] = {}
        for gt in labels:
            label_map.setdefault(gt.query_uuid, {})[gt.paragraph_uuid] = gt.label_score
        pred_map: Dict[str, List[Prediction]] = {}
        for pred in predictions:
            pred_map.setdefault(pred.query_uuid, []).append(pred)
        per_query: List[float] = []
        for query_id, preds in pred_map.items():
            preds.sort(key=lambda p: p.score, reverse=True)
            gains = []
            for rank, pred in enumerate(preds[: self.k], 1):
                rel = label_map.get(query_id, {}).get(pred.paragraph_uuid, 0)
                gains.append(rel / math.log2(rank + 1))
            ideal_rels = sorted(label_map.get(query_id, {}).values(), reverse=True)
            ideal = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels[: self.k]))
            per_query.append(sum(gains) / ideal if ideal > 0 else 0.0)
        return float(sum(per_query) / max(1, len(per_query)))


class RecallEvaluator:
    def __init__(self, k_list: Sequence[int]):
        self.k_list = list(k_list)

    def compute(self, predictions: Sequence[Prediction], labels: Sequence[GroundTruth]) -> Dict[str, float]:
        label_map: Dict[str, set[str]] = {}
        for gt in labels:
            label_map.setdefault(gt.query_uuid, set()).add(gt.paragraph_uuid)
        pred_map: Dict[str, List[Prediction]] = {}
        for pred in predictions:
            pred_map.setdefault(pred.query_uuid, []).append(pred)
        recall: Dict[str, float] = {}
        for k in self.k_list:
            hits = []
            for query_id, preds in pred_map.items():
                preds.sort(key=lambda p: p.score, reverse=True)
                picked = {pred.paragraph_uuid for pred in preds[:k]}
                relevant = label_map.get(query_id, set())
                hits.append(1.0 if picked & relevant else 0.0)
            recall[f"recall@{k}"] = float(sum(hits) / max(1, len(hits)))
        return recall


STATEFUL_CLASSES = [BenchmarkEncoder, SimpleVectorIndex, SimpleBM25Index, LightweightReranker]


def configure_stateful_hints(enabled: bool) -> None:
    for cls in STATEFUL_CLASSES:
        if enabled:
            setattr(cls, "__daft_hint__", "@daft.cls")
            setattr(cls, "__daft_stateful__", True)
        else:
            for attr in ("__daft_hint__", "__daft_stateful__"):
                if hasattr(cls, attr):
                    delattr(cls, attr)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


@dataclass
class BackendResult:
    label: str
    durations: List[float]
    result_snapshot: Optional[Dict[str, Any]] = None
    note: str = ""
    success: bool = True

    @property
    def best(self) -> float:
        return min(self.durations) if self.durations else float("nan")

    @property
    def mean(self) -> float:
        return statistics.mean(self.durations) if self.durations else float("nan")

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.durations) if len(self.durations) > 1 else 0.0

    def describe(self) -> str:
        if not self.success:
            return f"{self.label:<32} | unavailable ({self.note})"
        return (
            f"{self.label:<32} | mean={self.mean:.3f}s best={self.best:.3f}s "
            f"stdev={self.stdev:.3f}s"
        )


def benchmark(label: str, fn: Callable[[], Dict[str, Any]], repeats: int, warmup: int) -> BackendResult:
    for _ in range(warmup):
        fn()
    durations: List[float] = []
    snapshot: Optional[Dict[str, Any]] = None
    for _ in range(repeats):
        start = time.perf_counter()
        snapshot = fn()
        durations.append(time.perf_counter() - start)
    return BackendResult(label=label, durations=durations, result_snapshot=snapshot)


# ---------------------------------------------------------------------------
# Native Daft pipeline
# ---------------------------------------------------------------------------


def run_native_daft(dataset: SyntheticDataset, cfg: BenchmarkConfig) -> Dict[str, Any]:
    if daft is None:
        raise RuntimeError("Daft is not installed")

    # Encode passages inside Daft to reuse @daft.cls state
    passages_df = daft.from_pydict({
        "uuid": [p.uuid for p in dataset.passages],
        "text": [p.text for p in dataset.passages],
    })

    @daft.cls
    class NativePassageEncoder:
        def __init__(self, dim: int, seed: int):
            self.encoder = BenchmarkEncoder(dim=dim, seed=seed)

        @daft.method(return_dtype=daft.DataType.python())
        def __call__(self, uuid: str, text: str) -> EncodedPassage:
            embedding = self.encoder.encode(text, is_query=False)
            return EncodedPassage(uuid=uuid, text=text, embedding=embedding)

    passage_encoder = NativePassageEncoder(cfg.embedding_dim, cfg.encoder_seed)
    passages_df = passages_df.with_column(
        "encoded_passage",
        passage_encoder(passages_df["uuid"], passages_df["text"]),
    )
    encoded_passages = passages_df.collect().to_pydict()["encoded_passage"]

    # Build stateful components once
    vector_idx = SimpleVectorIndex(encoded_passages)
    bm25_idx = SimpleBM25Index(dataset.passages)
    reranker = LightweightReranker()

    @daft.cls
    class NativeQueryEncoder:
        def __init__(self, dim: int, seed: int):
            self.encoder = BenchmarkEncoder(dim=dim, seed=seed)

        @daft.method(return_dtype=daft.DataType.python())
        def __call__(self, uuid: str, text: str) -> EncodedQuery:
            embedding = self.encoder.encode(text, is_query=True)
            return EncodedQuery(uuid=uuid, text=text, embedding=embedding)

    @daft.cls
    class NativeColbertRetriever:
        def __init__(self, index: SimpleVectorIndex, top_k: int):
            self.index = index
            self.top_k = top_k

        @daft.method(return_dtype=daft.DataType.python())
        def __call__(self, encoded_query: EncodedQuery) -> List[SearchHit]:
            return self.index.search(encoded_query.embedding, self.top_k)

    @daft.cls
    class NativeBM25Retriever:
        def __init__(self, index: SimpleBM25Index, top_k: int):
            self.index = index
            self.top_k = top_k

        @daft.method(return_dtype=daft.DataType.python())
        def __call__(self, query_text: str) -> List[SearchHit]:
            return self.index.search(query_text, self.top_k)

    @daft.cls
    class NativeFuser:
        def __init__(self, k: int):
            self.rrf = RRFFusion(k)

        @daft.method(return_dtype=daft.DataType.python())
        def __call__(self, colbert_hits: List[SearchHit], bm25_hits: List[SearchHit]) -> List[SearchHit]:
            return self.rrf.fuse([colbert_hits, bm25_hits])

    @daft.cls
    class NativeReranker:
        def __init__(self, encoded_passages: Sequence[EncodedPassage], rerank_k: int):
            self.encoded = list(encoded_passages)
            self.rerank_k = rerank_k
            self.impl = LightweightReranker()

        @daft.method(return_dtype=daft.DataType.python())
        def __call__(self, query: Query, fused_hits: List[SearchHit]) -> List[SearchHit]:
            return self.impl.rerank(query, fused_hits, self.rerank_k, self.encoded)

    @daft.func(return_dtype=daft.DataType.python())
    def make_query(uuid: str, text: str) -> Query:
        return Query(uuid=uuid, text=text)

    @daft.func(return_dtype=daft.DataType.python())
    def hits_to_predictions(query: Query, hits: List[SearchHit]) -> List[Prediction]:
        return [Prediction(query_uuid=query.uuid, paragraph_uuid=hit.passage_uuid, score=hit.score) for hit in hits]

    queries_df = daft.from_pydict({
        "uuid": [q.uuid for q in dataset.queries],
        "text": [q.text for q in dataset.queries],
    })

    queries_df = queries_df.with_column("query", make_query(queries_df["uuid"], queries_df["text"]))
    query_encoder = NativeQueryEncoder(cfg.embedding_dim, cfg.encoder_seed)
    colbert = NativeColbertRetriever(vector_idx, cfg.top_k)
    bm25 = NativeBM25Retriever(bm25_idx, cfg.top_k)
    fuser = NativeFuser(cfg.rrf_k)
    native_reranker = NativeReranker(encoded_passages, cfg.rerank_k)

    queries_df = queries_df.with_column("encoded_query", query_encoder(queries_df["uuid"], queries_df["text"]))
    queries_df = queries_df.with_column("colbert_hits", colbert(queries_df["encoded_query"]))
    queries_df = queries_df.with_column("bm25_hits", bm25(queries_df["text"]))
    queries_df = queries_df.with_column(
        "fused_hits",
        fuser(queries_df["colbert_hits"], queries_df["bm25_hits"]),
    )
    queries_df = queries_df.with_column(
        "reranked_hits",
        native_reranker(queries_df["query"], queries_df["fused_hits"]),
    )
    queries_df = queries_df.with_column(
        "predictions",
        hits_to_predictions(queries_df["query"], queries_df["reranked_hits"]),
    )

    collected = queries_df.select("predictions").collect().to_pydict()["predictions"]
    flattened = [pred for per_query in collected for pred in per_query]
    evaluator = NDCGEvaluator(cfg.ndcg_k)
    recall_eval = RecallEvaluator(cfg.recall_k)
    ndcg = evaluator.compute(flattened, dataset.ground_truths)
    recall = recall_eval.compute(flattened, dataset.ground_truths)
    return {
        "evaluation_results": {
            "ndcg": ndcg,
            "ndcg_k": cfg.ndcg_k,
            "recall_metrics": recall,
        }
    }


# ---------------------------------------------------------------------------
# Hypernodes pipeline definition (mirrors the user script)
# ---------------------------------------------------------------------------


@node(output_name="passages")
def load_passages(passage_records: List[Dict[str, str]]) -> List[Passage]:
    return [Passage(**record) for record in passage_records]


@node(output_name="queries")
def load_queries(query_records: List[Dict[str, str]]) -> List[Query]:
    return [Query(**record) for record in query_records]


@node(output_name="ground_truths")
def load_ground_truths(ground_truth_records: List[Dict[str, Any]]) -> List[GroundTruth]:
    return [GroundTruth(**record) for record in ground_truth_records]


@node(output_name="encoder")
def create_encoder(embedding_dim: int, encoder_seed: int) -> BenchmarkEncoder:
    return BenchmarkEncoder(dim=embedding_dim, seed=encoder_seed)


@node(output_name="rrf")
def create_rrf(rrf_k: int) -> RRFFusion:
    return RRFFusion(k=rrf_k)


@node(output_name="ndcg_evaluator")
def create_ndcg(ndcg_k: int) -> NDCGEvaluator:
    return NDCGEvaluator(k=ndcg_k)


@node(output_name="recall_evaluator")
def create_recall(recall_k_list: List[int]) -> RecallEvaluator:
    return RecallEvaluator(k_list=recall_k_list)


@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[EncodedPassage]) -> SimpleVectorIndex:
    return SimpleVectorIndex(encoded_passages)


@node(output_name="bm25_index")
def build_bm25_index(passages: List[Passage]) -> SimpleBM25Index:
    return SimpleBM25Index(passages)


@node(output_name="reranker")
def create_reranker(encoder_seed: int) -> LightweightReranker:
    rng = np.random.default_rng(encoder_seed)
    bias = float(rng.uniform(0.02, 0.08))
    return LightweightReranker(bias=bias)


@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: BenchmarkEncoder) -> EncodedPassage:
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)


@node(output_name="encoded_query")
def encode_query(query: Query, encoder: BenchmarkEncoder) -> EncodedQuery:
    embedding = encoder.encode(query.text, is_query=True)
    return EncodedQuery(uuid=query.uuid, text=query.text, embedding=embedding)


@node(output_name="query")
def extract_query(encoded_query: EncodedQuery) -> Query:
    return Query(uuid=encoded_query.uuid, text=encoded_query.text)


@node(output_name="colbert_hits")
def retrieve_colbert(encoded_query: EncodedQuery, vector_index: SimpleVectorIndex, top_k: int) -> List[SearchHit]:
    return vector_index.search(encoded_query.embedding, top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(query: Query, bm25_index: SimpleBM25Index, top_k: int) -> List[SearchHit]:
    return bm25_index.search(query.text, top_k)


@node(output_name="fused_hits")
def fuse_hits(colbert_hits: List[SearchHit], bm25_hits: List[SearchHit], rrf: RRFFusion) -> List[SearchHit]:
    return rrf.fuse([colbert_hits, bm25_hits])


@node(output_name="reranked_hits")
def rerank_hits(
    query: Query,
    fused_hits: List[SearchHit],
    reranker: LightweightReranker,
    encoded_passages: List[EncodedPassage],
    rerank_k: int,
) -> List[SearchHit]:
    return reranker.rerank(query, fused_hits, rerank_k, encoded_passages)


@node(output_name="predictions")
def convert_predictions(query: Query, reranked_hits: List[SearchHit]) -> List[Prediction]:
    return [Prediction(query_uuid=query.uuid, paragraph_uuid=hit.passage_uuid, score=hit.score) for hit in reranked_hits]


@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[Prediction]]) -> List[Prediction]:
    return [pred for per_query in all_query_predictions for pred in per_query]


@node(output_name="ndcg_score")
def compute_ndcg(all_predictions: List[Prediction], ground_truths: List[GroundTruth], ndcg_evaluator: NDCGEvaluator) -> float:
    return ndcg_evaluator.compute(all_predictions, ground_truths)


@node(output_name="recall_metrics")
def compute_recall_metrics(all_predictions: List[Prediction], ground_truths: List[GroundTruth], recall_evaluator: RecallEvaluator) -> Dict[str, float]:
    return recall_evaluator.compute(all_predictions, ground_truths)


@node(output_name="evaluation_results")
def combine_results(ndcg_score: float, recall_metrics: Dict[str, float], ndcg_k: int) -> Dict[str, Any]:
    return {"ndcg": ndcg_score, "ndcg_k": ndcg_k, "recall_metrics": recall_metrics}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_retrieval_pipeline() -> Pipeline:
    encode_single_passage = Pipeline(nodes=[encode_passage], name="encode_single_passage")
    encode_single_query = Pipeline(nodes=[encode_query], name="encode_single_query")
    retrieve_single_query = Pipeline(
        nodes=[extract_query, retrieve_colbert, retrieve_bm25, fuse_hits, rerank_hits, convert_predictions],
        name="retrieve_single_query",
    )

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

    return Pipeline(
        nodes=[
            load_passages,
            load_queries,
            load_ground_truths,
            create_encoder,
            create_rrf,
            create_ndcg,
            create_recall,
            encode_passages_mapped,
            build_vector_index,
            build_bm25_index,
            create_reranker,
            encode_queries_mapped,
            retrieve_queries_mapped,
            flatten_predictions,
            compute_ndcg,
            compute_recall_metrics,
            combine_results,
        ],
        name="synthetic_hebrew_retrieval",
    )


def build_inputs(dataset: SyntheticDataset, cfg: BenchmarkConfig) -> Dict[str, Any]:
    return {
        "passage_records": [p.model_dump() for p in dataset.passages],
        "query_records": [q.model_dump() for q in dataset.queries],
        "ground_truth_records": [gt.model_dump() for gt in dataset.ground_truths],
        "embedding_dim": cfg.embedding_dim,
        "encoder_seed": cfg.encoder_seed,
        "rrf_k": cfg.rrf_k,
        "ndcg_k": cfg.ndcg_k,
        "recall_k_list": list(cfg.recall_k),
        "top_k": cfg.top_k,
        "rerank_k": cfg.rerank_k,
    }


def hypernodes_runner(engine_factory: Callable[[], Any], inputs: Dict[str, Any]) -> Callable[[], Dict[str, Any]]:
    def _run() -> Dict[str, Any]:
        pipeline = build_retrieval_pipeline().with_engine(engine_factory())
        try:
            return pipeline.run(inputs=inputs, output_name="evaluation_results")
        finally:
            engine = pipeline.effective_engine
            if hasattr(engine, "shutdown"):
                engine.shutdown(wait=True)
    return _run


def daft_engine_runner(engine_kwargs: Dict[str, Any], inputs: Dict[str, Any]) -> Callable[[], Dict[str, Any]]:
    if DaftEngine is None:
        raise RuntimeError("DaftEngine is unavailable (install daft)")

    def factory() -> DaftEngine:
        return DaftEngine(**engine_kwargs)

    return hypernodes_runner(factory, inputs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SCALE_TO_SIZES = {
    "tiny": (400, 80),
    "small": (1500, 300),
    "medium": (4000, 600),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Hypernodes vs Daft for retrieval")
    parser.add_argument("--scale", choices=SCALE_TO_SIZES.keys(), default="tiny")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--hypernodes-map-executor",
        choices=["sequential", "threaded", "parallel"],
        default="threaded",
        help="Executor for HypernodesEngine map()",
    )
    parser.add_argument("--inject-stateful-hints", action="store_true", help="Add __daft_hint__ to reusable classes")
    parser.add_argument("--show-plan", action="store_true", help="Show DaftEngine execution plans")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_passages, num_queries = SCALE_TO_SIZES[args.scale]
    dataset = build_dataset(num_passages, num_queries)
    cfg = BenchmarkConfig()
    inputs = build_inputs(dataset, cfg)

    results: List[BackendResult] = []

    if daft is not None:
        native_runner = lambda: run_native_daft(dataset, cfg)
        results.append(benchmark("Daft native (@daft.cls)", native_runner, args.repeats, args.warmup))
    else:
        results.append(BackendResult(label="Daft native (@daft.cls)", durations=[], success=False, note="daft missing"))

    hn_threaded = hypernodes_runner(
        lambda: HypernodesEngine(
            node_executor="threaded",
            map_executor=args.hypernodes_map_executor,
        ),
        inputs,
    )
    results.append(benchmark("HypernodesEngine(threaded)", hn_threaded, args.repeats, args.warmup))

    def make_daft_engine_kwargs(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kwargs = {"show_plan": args.show_plan}
        if extra:
            kwargs.update(extra)
        return kwargs

    configure_stateful_hints(False)
    try:
        default_daft_runner = daft_engine_runner(make_daft_engine_kwargs(), inputs)
        results.append(benchmark("DaftEngine(auto)", default_daft_runner, args.repeats, args.warmup))
    except Exception as exc:
        results.append(BackendResult(label="DaftEngine(auto)", durations=[], success=False, note=str(exc)))

    if args.inject_stateful_hints:
        configure_stateful_hints(True)
        try:
            hinted_runner = daft_engine_runner(make_daft_engine_kwargs(), inputs)
            results.append(benchmark("DaftEngine(+@daft.cls hints)", hinted_runner, args.repeats, args.warmup))
        except Exception as exc:
            results.append(
                BackendResult(label="DaftEngine(+@daft.cls hints)", durations=[], success=False, note=str(exc))
            )
        finally:
            configure_stateful_hints(False)

    print("\n=== Retrieval Benchmark Results ===")
    for result in results:
        print(result.describe())
    printable = [r for r in results if r.success and r.result_snapshot]
    if printable:
        sample = printable[0].result_snapshot or {}
        print("\nSample evaluation output:")
        print(sample)


if __name__ == "__main__":
    main()

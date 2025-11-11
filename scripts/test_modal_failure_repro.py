#!/usr/bin/env python3
"""Mock Hebrew retrieval pipeline that reproduces the Modal/Daft failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import modal
from pydantic import BaseModel

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine, fix_script_classes_for_modal


# =============================================================================
# Data models (mirror models.py from the real project)
# =============================================================================
class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: List[float]
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class Query(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedQuery(BaseModel):
    uuid: str
    text: str
    embedding: List[float]
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class GroundTruth(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    label_score: int
    model_config = {"frozen": True}


class SearchHit(BaseModel):
    passage_uuid: str
    score: float
    model_config = {"frozen": True}


class Prediction(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    score: float
    model_config = {"frozen": True}


# =============================================================================
# Stateful helpers (names match real implementation classes)
# =============================================================================
class ColBERTEncoder:
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = False
    __daft_max_concurrency__ = 2
    __daft_gpus__ = 0

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code

    def encode(self, text: str, is_query: bool = False) -> List[float]:
        base = len(text) + (5 if is_query else 0)
        return [float(base), float(base // 2), float(base // 3)]


class PLAIDIndex:
    __daft_hint__ = "@daft.cls"

    def __init__(self, encoded_passages: List[EncodedPassage], *_args, **_kwargs):
        self._passages = encoded_passages

    def search(self, query_embedding: List[float], k: int) -> List[SearchHit]:
        hits = []
        for idx, passage in enumerate(self._passages, 1):
            score = sum(query_embedding) - idx + len(passage.text)
            hits.append(SearchHit(passage_uuid=passage.uuid, score=float(score)))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]


class BM25IndexImpl:
    __daft_hint__ = "@daft.cls"

    def __init__(self, passages: List[Passage]):
        self._passages = passages

    def search(self, query_text: str, k: int) -> List[SearchHit]:
        hits = []
        for idx, passage in enumerate(self._passages, 1):
            score = float(len(query_text) + len(passage.text) - idx)
            hits.append(SearchHit(passage_uuid=passage.uuid, score=score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]


class ColBERTReranker:
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = False

    def __init__(self, encoder: ColBERTEncoder, passage_lookup: Dict[str, EncodedPassage]):
        self._encoder = encoder
        self._passages = passage_lookup

    def rerank(self, query: Query, candidates: List[SearchHit], k: int) -> List[SearchHit]:
        scores: Dict[str, float] = {}
        base = sum(self._encoder.encode(query.text, is_query=True))
        for idx, hit in enumerate(candidates[:k], 1):
            extra = sum(self._passages[hit.passage_uuid].embedding)
            scores[hit.passage_uuid] = base + extra - idx
        return [
            SearchHit(passage_uuid=uuid, score=score)
            for uuid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ]


class RRFFusion:
    __daft_hint__ = "@daft.cls"

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, lists: List[List[SearchHit]]) -> List[SearchHit]:
        scores: Dict[str, float] = {}
        for hits in lists:
            for rank, hit in enumerate(hits, 1):
                scores[hit.passage_uuid] = scores.get(hit.passage_uuid, 0.0) + 1 / (
                    self.k + rank
                )
        return [
            SearchHit(passage_uuid=uuid, score=score)
            for uuid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ]


class NDCGEvaluator:
    __daft_hint__ = "@daft.cls"

    def __init__(self, k: int):
        self.k = k

    def compute(self, predictions: List[Prediction], ground_truths: List[GroundTruth]) -> float:
        if not predictions:
            return 0.0
        top = sorted(predictions, key=lambda p: p.score, reverse=True)[: self.k]
        return sum(hit.score for hit in top) / len(top)


class RecallEvaluator:
    """Intentionally missing __daft_hint__ (matches failing scenario)."""

    def __init__(self, k_list: List[int]):
        self.k_list = k_list

    def compute(self, predictions: List[Prediction], ground_truths: List[GroundTruth]) -> Dict[str, float]:
        metrics = {}
        total = max(1, len(ground_truths))
        for k in self.k_list:
            metrics[f"recall@{k}"] = min(len(predictions), k) / total
        return metrics


# =============================================================================
# Fix all script classes for Modal/Daft serialization
# =============================================================================
fix_script_classes_for_modal()


# =============================================================================
# Nodes (copied from the real script, simplified to mock data)
# =============================================================================
@node(output_name="passages")
def load_passages(corpus_path: str) -> List[Passage]:
    return [Passage(uuid=f"p{i}", text=f"passage {i}") for i in range(5)]


@node(output_name="queries")
def load_queries(examples_path: str) -> List[Query]:
    return [Query(uuid=f"q{i}", text=f"query {i}") for i in range(2)]


@node(output_name="ground_truths")
def load_ground_truths(examples_path: str) -> List[GroundTruth]:
    return [GroundTruth(query_uuid="q0", paragraph_uuid="p0", label_score=1)]


@node(output_name="vector_index")
def build_vector_index(
    encoded_passages: List[EncodedPassage],
    index_folder: str,
    index_name: str,
    override: bool,
) -> PLAIDIndex:
    return PLAIDIndex(encoded_passages, index_folder, index_name, override)


@node(output_name="bm25_index")
def build_bm25_index(passages: List[Passage]) -> BM25IndexImpl:
    return BM25IndexImpl(passages)


@node(output_name="passage_lookup")
def build_passage_lookup(encoded_passages: List[EncodedPassage]) -> Dict[str, EncodedPassage]:
    return {p.uuid: p for p in encoded_passages}


@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: ColBERTEncoder) -> EncodedPassage:
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)


@node(output_name="encoded_query")
def encode_query(query: Query, encoder: ColBERTEncoder) -> EncodedQuery:
    embedding = encoder.encode(query.text, is_query=True)
    return EncodedQuery(uuid=query.uuid, text=query.text, embedding=embedding)


@node(output_name="query")
def extract_query(encoded_query: EncodedQuery) -> Query:
    return Query(uuid=encoded_query.uuid, text=encoded_query.text)


@node(output_name="colbert_hits")
def retrieve_colbert(encoded_query: EncodedQuery, vector_index: PLAIDIndex, top_k: int) -> List[SearchHit]:
    return vector_index.search(encoded_query.embedding, k=top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(query: Query, bm25_index: BM25IndexImpl, top_k: int) -> List[SearchHit]:
    return bm25_index.search(query.text, k=top_k)


@node(output_name="fused_hits")
def fuse_results(colbert_hits: List[SearchHit], bm25_hits: List[SearchHit], rrf: RRFFusion) -> List[SearchHit]:
    return rrf.fuse([colbert_hits, bm25_hits])


@node(output_name="reranked_hits")
def rerank_results(
    query: Query,
    fused_hits: List[SearchHit],
    encoder: ColBERTEncoder,
    passage_lookup: Dict[str, EncodedPassage],
    rerank_k: int,
) -> List[SearchHit]:
    reranker = ColBERTReranker(encoder, passage_lookup)
    return reranker.rerank(query, fused_hits, rerank_k)


@node(output_name="predictions")
def hits_to_predictions(query: Query, reranked_hits: List[SearchHit]) -> List[Prediction]:
    return [
        Prediction(query_uuid=query.uuid, paragraph_uuid=hit.passage_uuid, score=hit.score)
        for hit in reranked_hits
    ]


@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[Prediction]]) -> List[Prediction]:
    return [pred for group in all_query_predictions for pred in group]


@node(output_name="ndcg_score")
def compute_ndcg(
    all_predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    ndcg_evaluator: NDCGEvaluator,
) -> float:
    return ndcg_evaluator.compute(all_predictions, ground_truths)


@node(output_name="recall_metrics")
def compute_recall(
    all_predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    recall_evaluator: RecallEvaluator,
) -> Dict[str, float]:
    return recall_evaluator.compute(all_predictions, ground_truths)


@node(output_name="evaluation_results")
def combine_evaluation_results(
    ndcg_score: float,
    recall_metrics: Dict[str, float],
    ndcg_k: int,
) -> Dict[str, Any]:
    return {"ndcg": ndcg_score, "ndcg_k": ndcg_k, **recall_metrics}


# =============================================================================
# Pipelines + mapped nodes (identical structure to test_modal.py)
# =============================================================================
encode_single_passage = Pipeline(nodes=[encode_passage], name="encode_single_passage")
encode_single_query = Pipeline(nodes=[encode_query], name="encode_single_query")
retrieve_single_query = Pipeline(
    nodes=[
        extract_query,
        retrieve_colbert,
        retrieve_bm25,
        fuse_results,
        rerank_results,
        hits_to_predictions,
    ],
    name="retrieve_single_query",
)

encode_passages_mapped = encode_single_passage.as_node(
    input_mapping={"passages": "passage", "encoder": "encoder"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_passages_mapped",
)

encode_queries_mapped = encode_single_query.as_node(
    input_mapping={"queries": "query", "encoder": "encoder"},
    output_mapping={"encoded_query": "encoded_queries"},
    map_over="queries",
    name="encode_queries_mapped",
)

retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={
        "encoded_queries": "encoded_query",
        "vector_index": "vector_index",
        "passages": "passages",
        "top_k": "top_k",
        "rrf": "rrf",
        "encoder": "encoder",
        "rerank_k": "rerank_k",
        "passage_lookup": "passage_lookup",
    },
    output_mapping={"predictions": "all_query_predictions"},
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)

retrieval_pipeline = Pipeline(
    nodes=[
        load_passages,
        load_queries,
        load_ground_truths,
        encode_passages_mapped,
        build_vector_index,
        build_bm25_index,
        build_passage_lookup,
        encode_queries_mapped,
        retrieve_queries_mapped,
        flatten_predictions,
        compute_ndcg,
        compute_recall,
        combine_evaluation_results,
    ],
    name="mock_retrieval",
)


# =============================================================================
# Modal wrapper (matching real script)
# =============================================================================
app = modal.App("hypernodes-modal-failure-repro")
HYPERNODES_DIR = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .uv_pip_install("pydantic", "pyarrow", "daft")
    .add_local_dir(str(HYPERNODES_DIR), remote_path="/root/hypernodes")
)


@app.function(image=modal_image, timeout=600)
def run_pipeline(pipeline: Pipeline, inputs: dict, daft: bool = True) -> Dict[str, Any]:
    engine = DaftEngine(debug=True)
    pipeline = pipeline.with_engine(engine)
    return pipeline.run(inputs=inputs, output_name="evaluation_results")


@app.local_entrypoint()
def main():
    encoder = ColBERTEncoder(model_name="mock")
    rrf = RRFFusion(k=10)
    ndcg = NDCGEvaluator(k=2)
    recall = RecallEvaluator(k_list=[1, 2])

    print("Passage module:", Passage.__module__)
    print("RecallEvaluator module:", recall.__class__.__module__)

    inputs = {
        "corpus_path": "mock_corpus.parquet",
        "examples_path": "mock_examples.parquet",
        "model_name": "mock",
        "trust_remote_code": True,
        "index_folder": "index",
        "index_name": "mock",
        "override": True,
        "top_k": 2,
        "rerank_k": 2,
        "rrf_k": 60,
        "ndcg_k": 2,
        "recall_k_list": [1, 2],
        "encoder": encoder,
        "rrf": rrf,
        "ndcg_evaluator": ndcg,
        "recall_evaluator": recall,
    }

    result = run_pipeline.local(retrieval_pipeline, inputs, daft=True)
    print("Result:", result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hebrew Retrieval Pipeline - ULTRA FAST with DaftEngine

Key changes for DaftEngine compatibility:
1. Simple dicts instead of Pydantic models (faster, no serialization overhead)
2. Explicit DataType.python() for all nodes (Daft can handle it)
3. @daft.cls for lazy initialization
4. Batch encoding (97x speedup)

This version is optimized specifically for DaftEngine performance.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import daft
import numpy as np
import pandas as pd
import pytrec_eval
import torch
from daft import DataType, Series

from hypernodes import Pipeline, node
from hypernodes.telemetry import TelemetryCallback


def stateful(cls):
    """Mark a class as stateful for DaftEngine auto-detection."""
    setattr(cls, "__daft_stateful__", True)
    return cls


def format_seconds(seconds: float) -> str:
    return f"{seconds:0.2f}s"


# ==================== OPTIMIZED Encoder with @daft.cls ====================
@stateful
@daft.cls
class Model2VecEncoder:
    """Ultra-fast encoder with @daft.cls and batch support."""

    def __init__(self, model_name: str):
        from model2vec import StaticModel

        print(f"[Encoder] Loading {model_name}...")
        self._model = StaticModel.from_pretrained(model_name)

    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts, is_query: bool = False):
        """Batch encode - works with both lists and Series."""
        if isinstance(texts, Series):
            text_list = texts.to_pylist()
            batch_embeddings = self._model.encode(text_list)
            embeddings_list = [batch_embeddings[i] for i in range(len(text_list))]
            return Series.from_pylist(embeddings_list)
        else:
            batch_embeddings = self._model.encode(texts)
            return [batch_embeddings[i] for i in range(len(texts))]


# ==================== Simple Classes (No Pydantic!) ====================
@stateful
class CosineSimIndex:
    """Cosine similarity vector index."""

    def __init__(self, encoded_passages: List[dict]):
        self._passage_uuids = [p["uuid"] for p in encoded_passages]
        self._embeddings = np.vstack([p["embedding"] for p in encoded_passages])

    def search(self, query_embedding: np.ndarray, k: int) -> List[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity([query_embedding], self._embeddings)[0]
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [
            {"passage_uuid": self._passage_uuids[idx], "score": float(scores[idx])}
            for idx in top_k_indices
        ]


@stateful
class BM25IndexImpl:
    """BM25 index."""

    def __init__(self, passages: List[dict]):
        from rank_bm25 import BM25Okapi

        self._passage_uuids = [p["uuid"] for p in passages]
        tokenized_corpus = [p["text"].split() for p in passages]
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query_text: str, k: int) -> List[dict]:
        tokenized_query = query_text.split()
        scores = self._index.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [
            {"passage_uuid": self._passage_uuids[idx], "score": float(scores[idx])}
            for idx in top_k_indices
        ]


@stateful
class CrossEncoderReranker:
    """CrossEncoder reranker."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            model_name, device="cuda" if torch.cuda.is_available() else None
        )
        self._passage_lookup = None

    def rerank(
        self, query: dict, candidates: List[dict], k: int, encoded_passages: List[dict]
    ) -> List[dict]:
        if self._passage_lookup is None:
            self._passage_lookup = {p["uuid"]: p["text"] for p in encoded_passages}

        candidate_uuids = [hit["passage_uuid"] for hit in candidates[:k]]
        candidate_texts = [self._passage_lookup[uuid] for uuid in candidate_uuids]
        pairs = [(query["text"], text) for text in candidate_texts]
        scores = self._model.predict(pairs)
        reranked = sorted(
            zip(candidate_uuids, scores), key=lambda x: x[1], reverse=True
        )
        return [
            {"passage_uuid": uuid, "score": float(score)} for uuid, score in reranked
        ]

    def score_pairs(
        self,
        pairs: List[tuple[str, str]],
        batch_size: int,
    ) -> List[float]:
        """Score a list of (query, document) pairs in one call."""
        if not pairs:
            return []
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return scores.tolist()


@stateful
class PassthroughReranker:
    """Fallback reranker that keeps fused hits unchanged."""

    def rerank(
        self,
        query: dict,
        candidates: List[dict],
        k: int,
        encoded_passages: List[dict],
    ) -> List[dict]:
        return candidates[:k]


@stateful
class RRFFusion:
    """Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, results_list: List[List[dict]]) -> List[dict]:
        rrf_scores = {}
        for results in results_list:
            for rank, hit in enumerate(results, 1):
                rrf_scores[hit["passage_uuid"]] = rrf_scores.get(
                    hit["passage_uuid"], 0
                ) + 1 / (self.k + rank)
        sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"passage_uuid": uuid, "score": score} for uuid, score in sorted_hits]


@stateful
class NDCGEvaluator:
    """NDCG evaluation."""

    def __init__(self, k: int):
        self.k = k

    def compute(self, predictions: List[dict], ground_truths: List[dict]) -> float:
        pred_df = pd.DataFrame(predictions)
        gt_df = pd.DataFrame(ground_truths)

        qrels = {}
        for _, row in gt_df.iterrows():
            query_id, doc_id, relevance = (
                row["query_uuid"],
                row["paragraph_uuid"],
                int(row["label_score"]),
            )
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance

        results = {}
        for _, row in pred_df.iterrows():
            query_id, doc_id, score = (
                row["query_uuid"],
                row["paragraph_uuid"],
                float(row["score"]),
            )
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = score

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{self.k}"})
        scores = evaluator.evaluate(results)
        per_query_scores = [
            metrics[f"ndcg_cut_{self.k}"] for metrics in scores.values()
        ]
        return float(np.mean(per_query_scores))


@stateful
class RecallEvaluator:
    """Recall evaluation."""

    def __init__(self, k_list: List[int]):
        self.k_list = k_list

    def compute(self, predictions: List[dict], ground_truths: List[dict]) -> dict:
        pred_df = pd.DataFrame(predictions)
        gt_df = pd.DataFrame(ground_truths)

        recall_results = {}
        for k in self.k_list:
            top_k_list = []
            for query_uuid, group in pred_df.groupby("query_uuid"):
                top_k_group = group.nlargest(k, "score")
                top_k_list.append(top_k_group)
            top_k_preds = pd.concat(top_k_list, ignore_index=True)

            merged = gt_df.merge(
                top_k_preds, on=["query_uuid", "paragraph_uuid"], how="left"
            )
            recall_at_k = (
                merged[merged["label_score"] > 0]
                .groupby("query_uuid")["score"]
                .apply(lambda x: x.notnull().any())
                .mean()
            )
            recall_results[f"recall@{k}"] = recall_at_k

        return recall_results


# ==================== Data Loading Nodes ====================
@node(output_name="passages")
def load_passages(corpus_path: str, limit: int = 0) -> List[dict]:
    """Load passages as simple dicts."""
    if limit > 0:
        df = pd.read_parquet(corpus_path).head(limit)
    else:
        df = pd.read_parquet(corpus_path)
    return [{"uuid": row["uuid"], "text": row["passage"]} for _, row in df.iterrows()]


@node(output_name="queries")
def load_queries(examples_path: str) -> List[dict]:
    """Load queries as simple dicts."""
    df = pd.read_parquet(examples_path)
    query_df = df[["query_uuid", "query_text"]].drop_duplicates()
    return [
        {"uuid": row["query_uuid"], "text": row["query_text"]}
        for _, row in query_df.iterrows()
    ]


@node(output_name="ground_truths")
def load_ground_truths(examples_path: str) -> List[dict]:
    """Load ground truth labels as simple dicts."""
    df = pd.read_parquet(examples_path)
    df["label_score"] = df["label_score"].astype(int)
    return [
        {
            "query_uuid": row["query_uuid"],
            "paragraph_uuid": row["paragraph_uuid"],
            "label_score": row["label_score"],
        }
        for _, row in df.iterrows()
    ]


# ==================== BATCH ENCODING NODES ====================
@node(output_name="encoded_passages")
def encode_passages_batch(
    passages: List[dict], encoder: Model2VecEncoder
) -> List[dict]:
    """Encode ALL passages in ONE batch - 97x faster!"""
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts, is_query=False)
    return [
        {"uuid": p["uuid"], "text": p["text"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]


@node(output_name="encoded_queries")
def encode_queries_batch(queries: List[dict], encoder: Model2VecEncoder) -> List[dict]:
    """Encode ALL queries in ONE batch - 97x faster!"""
    texts = [q["text"] for q in queries]
    embeddings = encoder.encode_batch(texts, is_query=True)
    return [
        {"uuid": q["uuid"], "text": q["text"], "embedding": emb}
        for q, emb in zip(queries, embeddings)
    ]


# ==================== Index Building Nodes ====================
@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[dict]):
    """Build vector index."""
    return CosineSimIndex(encoded_passages)


@node(output_name="bm25_index")
def build_bm25_index(passages: List[dict]):
    """Build BM25 index."""
    return BM25IndexImpl(passages)


# ==================== Retrieval Nodes ====================
@node(output_name="query")
def extract_query(encoded_query: dict) -> dict:
    """Extract query from encoded query."""
    return {"uuid": encoded_query["uuid"], "text": encoded_query["text"]}


@node(output_name="vector_hits")
def retrieve_vector(encoded_query: dict, vector_index, top_k: int) -> List[dict]:
    """Retrieve from vector index."""
    return vector_index.search(encoded_query["embedding"], k=top_k)


@node(output_name="bm25_hits")
def retrieve_bm25(query: dict, bm25_index, top_k: int) -> List[dict]:
    """Retrieve from BM25 index."""
    return bm25_index.search(query["text"], k=top_k)


@node(output_name="fused_hits")
def fuse_results(
    vector_hits: List[dict], bm25_hits: List[dict], rrf: RRFFusion
) -> List[dict]:
    """Fuse vector and BM25 results."""
    return rrf.fuse([vector_hits, bm25_hits])


@node(output_name="all_query_predictions")
def batch_rerank_predictions(
    encoded_queries: List[dict],
    all_fused_hits: List[List[dict]],
    encoded_passages: List[dict],
    reranker: CrossEncoderReranker,
    rerank_k: int,
    reranker_batch_size: int = 128,
) -> List[List[dict]]:
    """Rerank all queries in batch using CrossEncoder predict."""
    # Fallback to per-query reranking if batching disabled
    if reranker_batch_size <= 0:
        per_query_predictions = []
        for query, fused_hits in zip(encoded_queries, all_fused_hits):
            hits = fused_hits or []
            reranked = reranker.rerank(
                query,
                hits,
                rerank_k,
                encoded_passages,
            )
            per_query_predictions.append(
                [
                    {
                        "query_uuid": query["uuid"],
                        "paragraph_uuid": hit["passage_uuid"],
                        "score": hit["score"],
                    }
                    for hit in reranked[:rerank_k]
                ]
            )
        return per_query_predictions

    passage_lookup = {p["uuid"]: p["text"] for p in encoded_passages}
    pairs: List[tuple[str, str]] = []
    metadata: List[tuple[str, str]] = []

    for query, fused_hits in zip(encoded_queries, all_fused_hits):
        hits = (fused_hits or [])[:rerank_k]
        for hit in hits:
            passage_text = passage_lookup.get(hit["passage_uuid"])
            if passage_text is None:
                continue
            pairs.append((query["text"], passage_text))
            metadata.append((query["uuid"], hit["passage_uuid"]))

    predictions_map: Dict[str, List[dict]] = {q["uuid"]: [] for q in encoded_queries}

    if pairs:
        scores = reranker.score_pairs(pairs, batch_size=reranker_batch_size)
        for (query_uuid, passage_uuid), score in zip(metadata, scores):
            predictions_map[query_uuid].append(
                {
                    "query_uuid": query_uuid,
                    "paragraph_uuid": passage_uuid,
                    "score": float(score),
                }
            )

    all_predictions: List[List[dict]] = []
    for query in encoded_queries:
        preds = predictions_map.get(query["uuid"], [])
        preds.sort(key=lambda x: x["score"], reverse=True)
        all_predictions.append(preds)
    return all_predictions


# ==================== Evaluation Nodes ====================
@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[dict]]) -> List[dict]:
    """Flatten nested predictions."""
    return [pred for query_preds in all_query_predictions for pred in query_preds]


@node(output_name="ndcg_score")
def compute_ndcg(
    all_predictions: List[dict],
    ground_truths: List[dict],
    ndcg_evaluator: NDCGEvaluator,
) -> float:
    """Compute NDCG score."""
    return ndcg_evaluator.compute(all_predictions, ground_truths)


@node(output_name="recall_metrics")
def compute_recall(
    all_predictions: List[dict],
    ground_truths: List[dict],
    recall_evaluator: RecallEvaluator,
) -> dict:
    """Compute Recall metrics."""
    return recall_evaluator.compute(all_predictions, ground_truths)


@node(output_name="evaluation_results")
def combine_evaluation_results(
    ndcg_score: float, recall_metrics: dict, ndcg_k: int
) -> dict:
    """Combine evaluation results."""
    return {"ndcg": ndcg_score, "ndcg_k": ndcg_k, "recall_metrics": recall_metrics}


# ==================== Pipeline Assembly ====================
retrieve_single_query = Pipeline(
    nodes=[
        extract_query,
        retrieve_vector,
        retrieve_bm25,
        fuse_results,
    ],
    name="retrieve_single_query",
)

retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={"encoded_queries": "encoded_query"},
    output_mapping={"fused_hits": "all_fused_hits"},
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)

# Initialize telemetry callback for profiling
# telemetry_callback = TelemetryCallback()  # Commented out - uncomment when needed

full_pipeline = Pipeline(
    nodes=[
        load_passages,
        load_queries,
        load_ground_truths,
        encode_passages_batch,
        build_vector_index,
        build_bm25_index,
        encode_queries_batch,
        retrieve_queries_mapped,
        batch_rerank_predictions,
        flatten_predictions,
        compute_ndcg,
        compute_recall,
        combine_evaluation_results,
    ],
    # callbacks=[ProgressCallback(), telemetry_callback],
    name="ultra_fast_retrieval",
)


def print_profiling_results(telemetry: TelemetryCallback) -> None:
    """Display profiling results from telemetry callback."""
    if not telemetry.span_data:
        print("\nNo profiling data available.")
        return

    print("\n" + "=" * 70)
    print("PROFILING RESULTS (Sequential Engine)")
    print("=" * 70)

    # Filter to only nodes (not pipelines or map operations)
    nodes = [s for s in telemetry.span_data if s.get("type") == "node"]

    if not nodes:
        print("No node data available.")
        return

    # Sort by duration (slowest first)
    nodes_sorted = sorted(nodes, key=lambda x: x["duration"], reverse=True)

    # Calculate total time from nodes
    total_node_time = sum(s["duration"] for s in nodes)

    print(f"\n{'Node Name':<40} {'Time':>12} {'% Total':>10} {'Status':>10}")
    print("-" * 70)

    for span in nodes_sorted:
        name = span["name"]
        duration = span["duration"]
        percentage = (duration / total_node_time * 100) if total_node_time > 0 else 0
        status = "⚡ cached" if span.get("cached", False) else ""

        print(
            f"{name:<40} {format_seconds(duration):>12} {percentage:>9.1f}% {status:>10}"
        )

    print("-" * 70)
    print(
        f"{'TOTAL (nodes only)':<40} {format_seconds(total_node_time):>12} {'100.0%':>10}"
    )

    # Show breakdown by category
    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN")
    print("=" * 70)

    categories = {
        "Data Loading": ["load_passages", "load_queries", "load_ground_truths"],
        "Encoding": ["encode_passages_batch", "encode_queries_batch"],
        "Index Building": ["build_vector_index", "build_bm25_index"],
        "Retrieval": [
            "retrieve_queries_mapped",
            "retrieve_single_query",
            "extract_query",
            "retrieve_vector",
            "retrieve_bm25",
            "fuse_results",
            "rerank_with_crossencoder",
            "hits_to_predictions",
        ],
        "Evaluation": [
            "flatten_predictions",
            "compute_ndcg",
            "compute_recall",
            "combine_evaluation_results",
        ],
    }

    category_times = {}
    for category, node_names in categories.items():
        total = sum(s["duration"] for s in nodes if s["name"] in node_names)
        if total > 0:
            category_times[category] = total

    # Sort by time
    for category, total in sorted(
        category_times.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (total / total_node_time * 100) if total_node_time > 0 else 0
        print(f"{category:<40} {format_seconds(total):>12} {percentage:>9.1f}%")

    print("=" * 70)

    # Show map operations if any
    map_ops = [s for s in telemetry.span_data if s.get("type") == "map"]
    if map_ops:
        print("\nMAP OPERATIONS")
        print("=" * 70)
        for span in map_ops:
            print(f"{span['name']:<40} {format_seconds(span['duration']):>12}")
        print("=" * 70)

    print(
        "\nNote: Run in Jupyter and call telemetry.get_waterfall_chart() for visual timeline"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hebrew retrieval pipeline with SeqEngine or DaftEngine."
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Dataset size (matches data/sample_<N>).",
    )
    dataset_group.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
        help="Parquet split to use.",
    )
    dataset_group.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit corpus size (0 = all passages).",
    )

    # Model options
    model_group = parser.add_argument_group("Models")
    model_group.add_argument(
        "--encoder-model",
        default="minishlab/potion-retrieval-32M",
        help="Encoder model checkpoint.",
    )
    model_group.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="CrossEncoder checkpoint.",
    )
    model_group.add_argument(
        "--disable-cross-encoder",
        action="store_true",
        help="Skip CrossEncoder reranking.",
    )

    # Retrieval options
    retrieval_group = parser.add_argument_group("Retrieval")
    retrieval_group.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="Candidates to retrieve per index.",
    )
    retrieval_group.add_argument(
        "--rerank-k",
        type=int,
        default=300,
        help="Candidates to rerank.",
    )
    retrieval_group.add_argument(
        "--reranker-batch-size",
        type=int,
        default=128,
        help="CrossEncoder batch size (<=0 to disable batching).",
    )
    retrieval_group.add_argument(
        "--ndcg-k",
        type=int,
        default=20,
        help="NDCG evaluation cutoff.",
    )

    # Engine options
    engine_group = parser.add_argument_group("Engine")
    engine_group.add_argument(
        "--engine",
        choices=["sequential", "dask", "daft"],
        default="sequential",
        help="Execution engine.",
    )
    engine_group.add_argument(
        "--daft-threaded-batch",
        action="store_true",
        help="Enable DaftEngine threaded batch UDFs.",
    )
    engine_group.add_argument(
        "--daft-batch-size",
        type=int,
        help="Override DaftEngine batch_size.",
    )
    engine_group.add_argument(
        "--daft-max-workers",
        type=int,
        help="Override DaftEngine max_workers.",
    )
    engine_group.add_argument(
        "--daft-max-concurrency",
        type=int,
        help="Max concurrent @daft.cls instances.",
    )
    engine_group.add_argument(
        "--daft-use-process",
        action="store_true",
        help="Use process isolation for Daft UDFs.",
    )
    engine_group.add_argument(
        "--daft-gpus",
        type=int,
        help="GPUs per @daft.cls instance.",
    )

    # Output options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logfire for local-only telemetry (no remote upload)
    try:
        import logfire

        logfire.configure(send_to_logfire=False)
    except ImportError:
        print("Warning: logfire not installed. Telemetry will be disabled.")
        print("Install with: pip install 'hypernodes[telemetry]'")

    dataset_dir = Path("data") / f"sample_{args.examples}"
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory '{dataset_dir}' not found.")

    corpus_path = dataset_dir / "corpus.parquet"
    examples_path = dataset_dir / f"{args.split}.parquet"

    # Instantiate heavy objects (stateful for Daft).
    encoder = Model2VecEncoder(args.encoder_model)
    rrf = RRFFusion(k=60)
    ndcg_evaluator = NDCGEvaluator(k=args.ndcg_k)
    recall_evaluator = RecallEvaluator(k_list=[20, 50, 100, 200, 300])
    reranker = (
        PassthroughReranker()
        if args.disable_cross_encoder
        else CrossEncoderReranker(model_name=args.reranker_model)
    )

    pipeline = full_pipeline
    engine_name = "SeqEngine"

    if args.engine == "daft":
        from hypernodes.engines import DaftEngine

        daft_config: Dict[str, Any] = {}
        if args.daft_batch_size:
            daft_config["batch_size"] = args.daft_batch_size
        if args.daft_max_workers:
            daft_config["max_workers"] = args.daft_max_workers
        if args.daft_max_concurrency:
            daft_config["max_concurrency"] = args.daft_max_concurrency
        if args.daft_use_process:
            daft_config["use_process"] = True
        if args.daft_gpus:
            daft_config["gpus"] = args.daft_gpus

        pipeline = pipeline.with_engine(
            DaftEngine(
                use_batch_udf=args.daft_threaded_batch,
                default_daft_config=daft_config or None,
            )
        )
        engine_name = "DaftEngine"
    elif args.engine == "dask":
        try:
            from hypernodes.engines import DaskEngine
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "DaskEngine requires the optional 'dask' dependency. "
                "Install with: pip install 'hypernodes[dask]'"
            ) from exc

        pipeline = pipeline.with_engine(DaskEngine())
        engine_name = "DaskEngine"
    else:
        from hypernodes.engines import SeqEngine

        pipeline = pipeline.with_engine(SeqEngine())

    if not args.quiet:
        print("=" * 70)
        print("ULTRA FAST RETRIEVAL PIPELINE")
        print("=" * 70)
        print(f"Engine: {engine_name}")
        print(
            f"Dataset: sample_{args.examples} ({args.split}) | "
            f"Top-K: {args.top_k} | Rerank-K: {args.rerank_k}"
        )
        if args.engine == "daft":
            print(
                f"Daft threaded batch: {'ON' if args.daft_threaded_batch else 'OFF'} | "
                f"Config: {daft_config if daft_config else 'auto'}"
            )
        print("Optimizations: batch encoding, @daft.cls, simple dict payloads")
        print("=" * 70)
        print("Running pipeline...\n")

    inputs = {
        "corpus_path": str(corpus_path),
        "limit": args.limit,
        "examples_path": str(examples_path),
        "encoder": encoder,
        "rrf": rrf,
        "ndcg_evaluator": ndcg_evaluator,
        "recall_evaluator": recall_evaluator,
        "reranker": reranker,
        "top_k": args.top_k,
        "rerank_k": args.rerank_k,
        "ndcg_k": args.ndcg_k,
        "reranker_batch_size": args.reranker_batch_size,
    }

    start_time = time.perf_counter()
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    elapsed = time.perf_counter() - start_time

    eval_results = results["evaluation_results"]

    if not args.quiet:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"NDCG@{eval_results['ndcg_k']}: {eval_results['ndcg']:.4f}")
        print("\nRecall Metrics:")
        for metric, value in eval_results["recall_metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print("=" * 70)
        print(f"Total time: {format_seconds(elapsed)}")
        print("=" * 70)

        # Display profiling results from telemetry
        # print_profiling_results(telemetry_callback)  # Commented out - enable telemetry_callback first


if __name__ == "__main__":
    main()

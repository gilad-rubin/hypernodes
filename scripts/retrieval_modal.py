#!/usr/bin/env python3
"""
Hebrew Retrieval Pipeline - Modal Deployment

Run the ultra-fast retrieval pipeline on Modal with GPU support.

Usage:
    # Sequential engine (CPU)
    uv run modal run scripts/retrieval_modal.py --examples 5 --split test

    # DaftEngine (GPU accelerated)
    uv run modal run scripts/retrieval_modal.py --engine daft --examples 10

    # With batch UDFs for better performance
    uv run modal run scripts/retrieval_modal.py --engine daft --daft-threaded-batch

    # Disable cross-encoder reranking for faster execution
    uv run modal run scripts/retrieval_modal.py --disable-cross-encoder

    # Larger dataset with different split
    uv run modal run scripts/retrieval_modal.py --examples 100 --split dev
"""

from pathlib import Path
from typing import Any, Dict

import modal

# Create Modal app
app = modal.App("hypernodes-hebrew-retrieval")

# Create Modal volumes (using same names as the working test_modal.py)
models_volume = modal.Volume.from_name("mafat-models", create_if_missing=True)
data_volume = modal.Volume.from_name("mafat-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("mafat-cache", create_if_missing=True)

# Get local paths
repo_root = Path(__file__).parent.parent
hypernodes_dir = repo_root / "src" / "hypernodes"
scripts_dir = repo_root / "scripts"

# Define Modal image with all dependencies (matches working test_modal.py)
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "HF_HOME": "/root/models",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
        }
    )
    .uv_pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "pytrec-eval",
        "pyarrow",
        "torch",
        "sentence-transformers",
        "model2vec",
        "rank-bm25",
        "rich",
        "tqdm",
        "pydantic",
        "ipywidgets",
        "python-dotenv",
        "networkx",
        "loky",
        "psutil",
        "diskcache",
        "graphviz",
        "daft",
    )
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
    .add_local_file(
        str(scripts_dir / "retrieval_ultra_fast.py"),
        remote_path="/root/retrieval_ultra_fast.py",
    )
)


@app.function(
    gpu="T4",  # Use T4 for cost-effectiveness, upgrade to A10G for more power
    image=modal_image,
    timeout=3600,  # 1 hour timeout
    volumes={
        "/root/models": models_volume,
        "/root/data": data_volume,
        "/cache": cache_volume,
    },
)
def run_retrieval_pipeline(
    pipeline_params: Dict[str, Any],
    engine_type: str = "sequential",
    daft_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run the retrieval pipeline on Modal.

    Args:
        pipeline_params: Pipeline input parameters
        engine_type: "sequential" or "daft"
        daft_config: Optional DaftEngine configuration

    Returns:
        Evaluation results dictionary
    """
    import sys
    import time

    # Ensure root directory is in path (retrieval_ultra_fast.py is at /root/)
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from hypernodes.cache import DiskCache
    from hypernodes.telemetry import ProgressCallback

    print(f"Starting retrieval pipeline with {engine_type} engine...")
    start_time = time.perf_counter()

    # Import all node definitions and classes from retrieval_ultra_fast
    from retrieval_ultra_fast import (
        CrossEncoderReranker,
        Model2VecEncoder,
        NDCGEvaluator,
        PassthroughReranker,
        RecallEvaluator,
        RRFFusion,
        full_pipeline,
    )

    encoder = Model2VecEncoder(pipeline_params["encoder_model"])
    rrf = RRFFusion(k=60)
    ndcg_evaluator = NDCGEvaluator(k=pipeline_params["ndcg_k"])
    recall_evaluator = RecallEvaluator(k_list=[20, 50, 100, 200, 300])
    reranker = (
        PassthroughReranker()
        if pipeline_params.get("disable_cross_encoder", False)
        else CrossEncoderReranker(model_name=pipeline_params["reranker_model"])
    )

    # Build inputs
    inputs = {
        "corpus_path": pipeline_params["corpus_path"],
        "limit": pipeline_params.get("limit", 0),
        "examples_path": pipeline_params["examples_path"],
        "encoder": encoder,
        "rrf": rrf,
        "ndcg_evaluator": ndcg_evaluator,
        "recall_evaluator": recall_evaluator,
        "reranker": reranker,
        "top_k": pipeline_params["top_k"],
        "rerank_k": pipeline_params["rerank_k"],
        "ndcg_k": pipeline_params["ndcg_k"],
        "reranker_batch_size": pipeline_params.get("reranker_batch_size", 128),
    }

    # Configure pipeline with engine
    pipeline = full_pipeline

    if engine_type == "daft":
        from hypernodes.engines import DaftEngine

        engine = DaftEngine(
            use_batch_udf=daft_config.get("use_batch_udf", False)
            if daft_config
            else False,
            default_daft_config=daft_config,
        )
        pipeline = pipeline.with_engine(engine)
    else:
        from hypernodes.engines import SequentialEngine

        pipeline = pipeline.with_engine(SequentialEngine())
        pipeline = pipeline.with_cache(DiskCache(path="/cache"))

    # Add progress callback
    pipeline = pipeline.with_callbacks([ProgressCallback()])

    # Run pipeline
    print("Executing pipeline...")
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)

    elapsed = time.perf_counter() - start_time

    eval_results = results["evaluation_results"]
    eval_results["elapsed_time"] = elapsed

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"NDCG@{eval_results['ndcg_k']}: {eval_results['ndcg']:.4f}")

    return eval_results


# Removed upload_data function - data is mounted directly in run_retrieval_pipeline


@app.local_entrypoint()
def main(
    examples: int = 5,
    split: str = "test",
    engine: str = "sequential",
    top_k: int = 300,
    rerank_k: int = 300,
    ndcg_k: int = 20,
    encoder_model: str = "minishlab/potion-retrieval-32M",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    disable_cross_encoder: bool = False,
    daft_threaded_batch: bool = False,
):
    """
    Local entrypoint for Modal execution.

    Usage:
        # Sequential engine
        uv run modal run scripts/retrieval_modal.py

        # DaftEngine with default config
        uv run modal run scripts/retrieval_modal.py --engine daft

        # DaftEngine with batch UDFs
        uv run modal run scripts/retrieval_modal.py --engine daft --daft-threaded-batch
    """
    from time import time

    # Build paths - use relative paths like the working test_modal.py
    # The data volume is mounted at /root/data, and working dir is /root
    # Build pipeline parameters
    pipeline_params = {
        "corpus_path": f"data/sample_{examples}/corpus.parquet",
        "examples_path": f"data/sample_{examples}/{split}.parquet",
        "encoder_model": encoder_model,
        "reranker_model": reranker_model,
        "top_k": top_k,
        "rerank_k": rerank_k,
        "ndcg_k": ndcg_k,
        "reranker_batch_size": 128,
        "disable_cross_encoder": disable_cross_encoder,
    }

    # Build daft config if using DaftEngine
    daft_config = None
    if engine == "daft":
        daft_config = {
            "use_batch_udf": daft_threaded_batch,
        }

    print("=" * 70)
    print("HEBREW RETRIEVAL PIPELINE - MODAL EXECUTION")
    print("=" * 70)
    print(f"Engine: {engine}")
    print(f"Dataset: sample_{examples} ({split})")
    print(f"Top-K: {top_k} | Rerank-K: {rerank_k} | NDCG-K: {ndcg_k}")
    if engine == "daft":
        print(f"Daft Config: {daft_config}")
    print("=" * 70)

    # Run on Modal
    start_time = time()
    results = run_retrieval_pipeline.remote(
        pipeline_params=pipeline_params,
        engine_type=engine,
        daft_config=daft_config,
    )
    end_time = time()

    print("\n" + "=" * 70)
    print("MODAL EXECUTION RESULTS")
    print("=" * 70)
    print(f"Total time (including overhead): {end_time - start_time:.2f}s")
    print(f"Pipeline execution time: {results['elapsed_time']:.2f}s")
    print(f"NDCG@{results['ndcg_k']}: {results['ndcg']:.4f}")
    print("\nRecall Metrics:")
    for metric, value in results["recall_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    # For local testing (runs inside Modal container)
    import sys

    print("Use 'uv run modal run scripts/retrieval_modal.py' to execute on Modal")
    sys.exit(1)

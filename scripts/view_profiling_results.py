#!/usr/bin/env python3
"""
View profiling results and generate waterfall chart.

This script re-runs the retrieval pipeline with telemetry enabled
and generates an interactive HTML waterfall chart.

Usage:
    python scripts/view_profiling_results.py --examples 5
"""

import argparse
from pathlib import Path

import logfire

from retrieval_ultra_fast import (
    CrossEncoderReranker,
    Model2VecEncoder,
    NDCGEvaluator,
    PassthroughReranker,
    RecallEvaluator,
    RRFFusion,
    full_pipeline,
)
from hypernodes.telemetry import TelemetryCallback, ProgressCallback


def main():
    parser = argparse.ArgumentParser(
        description="Profile the retrieval pipeline and generate waterfall chart."
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Dataset size variant (matches data/sample_<N>).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
        help="Which parquet split to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/profiling_waterfall.html",
        help="Output path for waterfall chart HTML.",
    )
    parser.add_argument(
        "--disable-cross-encoder",
        action="store_true",
        help="Skip CrossEncoder reranking for faster profiling.",
    )
    args = parser.parse_args()

    # Configure logfire
    logfire.configure(send_to_logfire=False)

    # Create fresh telemetry callback
    telemetry = TelemetryCallback()

    # Rebuild pipeline with telemetry
    pipeline = full_pipeline.with_callbacks([ProgressCallback(), telemetry])

    # Setup data paths
    dataset_dir = Path("data") / f"sample_{args.examples}"
    corpus_path = dataset_dir / "corpus.parquet"
    examples_path = dataset_dir / f"{args.split}.parquet"

    # Initialize components
    encoder = Model2VecEncoder("minishlab/potion-retrieval-32M")
    rrf = RRFFusion(k=60)
    ndcg_evaluator = NDCGEvaluator(k=20)
    recall_evaluator = RecallEvaluator(k_list=[20, 50, 100, 200, 300])
    reranker = (
        PassthroughReranker()
        if args.disable_cross_encoder
        else CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    inputs = {
        "corpus_path": str(corpus_path),
        "limit": 0,
        "examples_path": str(examples_path),
        "encoder": encoder,
        "rrf": rrf,
        "ndcg_evaluator": ndcg_evaluator,
        "recall_evaluator": recall_evaluator,
        "reranker": reranker,
        "top_k": 300,
        "rerank_k": 300,
        "ndcg_k": 20,
    }

    print("=" * 70)
    print("Running pipeline with profiling enabled...")
    print("=" * 70)

    # Run pipeline
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    eval_results = results["evaluation_results"]
    print(f"NDCG@{eval_results['ndcg_k']}: {eval_results['ndcg']:.4f}")

    # Generate waterfall chart
    print("\n" + "=" * 70)
    print("Generating waterfall chart...")
    print("=" * 70)

    chart = telemetry.get_waterfall_chart()

    # Save as HTML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.write_html(str(output_path))

    print(f"✓ Waterfall chart saved to: {output_path}")
    print(f"  Open it in your browser to view the interactive timeline!")

    # Print summary stats
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    nodes = [s for s in telemetry.span_data if s.get("type") == "node"]
    if nodes:
        total_time = sum(s["duration"] for s in nodes)
        print(f"\nTotal node execution time: {total_time:.2f}s")
        print(f"Number of nodes executed: {len(nodes)}")

        # Group by name and show aggregates
        from collections import defaultdict

        by_name = defaultdict(list)
        for node in nodes:
            by_name[node["name"]].append(node["duration"])

        print("\nTop 10 nodes by total time:")
        aggregated = [
            (name, sum(durations), len(durations), sum(durations) / len(durations))
            for name, durations in by_name.items()
        ]
        aggregated.sort(key=lambda x: x[1], reverse=True)

        for name, total, count, avg in aggregated[:10]:
            pct = (total / total_time * 100) if total_time > 0 else 0
            print(f"  {name:40s} {total:8.2f}s ({pct:5.1f}%) - {count} calls, {avg:.3f}s avg")


if __name__ == "__main__":
    main()






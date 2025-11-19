#!/usr/bin/env python3
"""
Benchmark: Original vs Optimized Retrieval Pipeline

Compares:
1. Original (map-based encoding)
2. Optimized (batch encoding + @stateful)

With both SeqEngine and DaftEngine
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_pipeline(pipeline, inputs, name, engine_name):
    """Benchmark a pipeline with timing."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name} + {engine_name}")
    print(f"{'=' * 60}")

    start_time = time.time()
    results = pipeline.run(output_name="evaluation_results", inputs=inputs)
    elapsed_time = time.time() - start_time

    eval_results = results["evaluation_results"]

    print(f"NDCG@{eval_results['ndcg_k']}: {eval_results['ndcg']:.4f}")
    print(f"Time: {elapsed_time:.2f}s")

    return {
        "name": name,
        "engine": engine_name,
        "time": elapsed_time,
        "ndcg": eval_results["ndcg"],
        "recall_metrics": eval_results["recall_metrics"],
    }


def main():
    print("\n" + "=" * 60)
    print("RETRIEVAL PIPELINE OPTIMIZATION BENCHMARK")
    print("=" * 60)

    # Import both pipelines
    print("\nLoading pipelines...")

    # Original pipeline
    from retrieval import inputs, pipeline

    original_pipeline = pipeline

    # Optimized pipeline
    from retrieval_optimized import inputs as inputs_opt
    from retrieval_optimized import pipeline_optimized

    # Use same inputs
    assert inputs == inputs_opt, "Inputs must match!"

    # Engines to test
    from hypernodes.engines import DaftEngine, SeqEngine

    engines = [
        ("SeqEngine", SeqEngine()),
        ("DaftEngine", DaftEngine()),
    ]

    results = []

    # Test original pipeline
    print("\n" + "=" * 60)
    print("ORIGINAL PIPELINE (map-based encoding)")
    print("=" * 60)

    for engine_name, engine in engines:
        test_pipeline = original_pipeline.with_engine(engine)
        result = benchmark_pipeline(test_pipeline, inputs, "Original", engine_name)
        results.append(result)
        time.sleep(1)  # Brief pause between runs

    # Test optimized pipeline
    print("\n" + "=" * 60)
    print("OPTIMIZED PIPELINE (batch encoding + @stateful)")
    print("=" * 60)

    for engine_name, engine in engines:
        test_pipeline = pipeline_optimized.with_engine(engine)
        result = benchmark_pipeline(test_pipeline, inputs, "Optimized", engine_name)
        results.append(result)
        time.sleep(1)  # Brief pause between runs

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Pipeline':<20} {'Engine':<20} {'Time (s)':<12} {'NDCG@20':<10}")
    print("-" * 62)

    for r in results:
        print(
            f"{r['name']:<20} {r['engine']:<20} {r['time']:<12.2f} {r['ndcg']:<10.4f}"
        )

    # Calculate speedups
    print("\n" + "=" * 60)
    print("SPEEDUPS")
    print("=" * 60)

    baseline = results[0]  # Original + Sequential

    for r in results[1:]:
        speedup = baseline["time"] / r["time"]
        print(f"{r['name']} + {r['engine']}: {speedup:.2f}x faster than baseline")

    # Find best
    best = min(results, key=lambda x: x["time"])
    print(f"\n✅ Best: {best['name']} + {best['engine']} = {best['time']:.2f}s")

    # Verify correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    ndcgs = [r["ndcg"] for r in results]
    if max(ndcgs) - min(ndcgs) < 0.001:
        print("✅ All pipelines produce identical results!")
    else:
        print("⚠️ WARNING: Results differ!")
        for r in results:
            print(f"  {r['name']} + {r['engine']}: NDCG = {r['ndcg']:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark retrieval encoding: Original vs Optimized with @daft.cls

This script compares:
1. Original: Without @daft.cls (eager loading, no batch optimization)
2. Optimized: With @daft.cls (lazy loading, batch optimization)
"""

import time
from typing import List

import numpy as np
from model2vec import StaticModel
import pandas as pd

import daft
from daft import DataType, Series

from hypernodes import Pipeline, node


# ==================== ORIGINAL Encoder (No optimization) ====================
class Model2VecEncoderOriginal:
    """Original encoder without optimizations."""
    
    def __init__(self, model_name: str):
        """Eager loading - model loaded immediately."""
        print(f"[Original] Loading model eagerly: {model_name}")
        start = time.time()
        self._model = StaticModel.from_pretrained(model_name)
        elapsed = time.time() - start
        print(f"[Original] Model loaded in {elapsed:.2f}s")
    
    def encode_batch(self, texts: List[str], is_query: bool = False) -> List[np.ndarray]:
        """Batch encode (works but not optimized for Daft)."""
        batch_embeddings = self._model.encode(texts)
        return [batch_embeddings[i] for i in range(len(texts))]


# ==================== OPTIMIZED Encoder with @daft.cls ====================
@daft.cls
class Model2VecEncoderOptimized:
    """Optimized encoder with @daft.cls."""
    
    def __init__(self, model_name: str):
        """Lazy loading - model loaded on first use."""
        print(f"[Optimized] Creating encoder (lazy): {model_name}")
        self.model_name = model_name
        self._model = StaticModel.from_pretrained(model_name)
        print(f"[Optimized] Model loaded")
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts, is_query: bool = False):
        """Batch encode with dual-mode support."""
        if isinstance(texts, Series):
            # Daft Series path
            text_list = texts.to_pylist()
            batch_embeddings = self._model.encode(text_list)
            embeddings_list = [batch_embeddings[i] for i in range(len(text_list))]
            return Series.from_pylist(embeddings_list)
        else:
            # Python list path
            batch_embeddings = self._model.encode(texts)
            return [batch_embeddings[i] for i in range(len(texts))]


# ==================== Nodes ====================
@node(output_name="passages")
def load_passages(corpus_path: str, limit: int = 0):
    """Load passages."""
    if limit > 0:
        df = pd.read_parquet(corpus_path).head(limit)
    else:
        df = pd.read_parquet(corpus_path)
    return [{"uuid": row["uuid"], "text": row["passage"]} for _, row in df.iterrows()]


@node(output_name="queries")
def load_queries(examples_path: str):
    """Load queries."""
    df = pd.read_parquet(examples_path)
    query_df = df[["query_uuid", "query_text"]].drop_duplicates()
    return [
        {"uuid": row["query_uuid"], "text": row["query_text"]}
        for _, row in query_df.iterrows()
    ]


@node(output_name="encoded_passages")
def encode_passages(passages, encoder):
    """Encode passages in batch."""
    print(f"\nEncoding {len(passages)} passages...")
    start = time.time()
    
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts, is_query=False)
    
    elapsed = time.time() - start
    print(f"Encoding time: {elapsed:.3f}s ({len(passages)/elapsed:.1f} passages/s)")
    
    return [
        {"uuid": p["uuid"], "text": p["text"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]


@node(output_name="encoded_queries")
def encode_queries(queries, encoder):
    """Encode queries in batch."""
    print(f"\nEncoding {len(queries)} queries...")
    start = time.time()
    
    texts = [q["text"] for q in queries]
    embeddings = encoder.encode_batch(texts, is_query=True)
    
    elapsed = time.time() - start
    print(f"Encoding time: {elapsed:.3f}s ({len(queries)/elapsed:.1f} queries/s)")
    
    return [
        {"uuid": q["uuid"], "text": q["text"], "embedding": emb}
        for q, emb in zip(queries, embeddings)
    ]


def run_benchmark(encoder_class, name: str, num_examples: int = 5):
    """Run benchmark with given encoder."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {name}")
    print("=" * 70)
    
    # Create encoder
    print(f"\n1. Creating encoder...")
    encoder_start = time.time()
    encoder = encoder_class("minishlab/potion-retrieval-32M")
    encoder_time = time.time() - encoder_start
    print(f"   Encoder creation time: {encoder_time:.3f}s")
    
    # Create pipeline
    pipeline = Pipeline(
        nodes=[
            load_passages,
            load_queries,
            encode_passages,
            encode_queries,
        ],
        name=f"encoding_benchmark_{name}",
    )
    
    inputs = {
        "corpus_path": f"data/sample_{num_examples}/corpus.parquet",
        "limit": 0,
        "examples_path": f"data/sample_{num_examples}/test.parquet",
        "encoder": encoder,
    }
    
    # Run pipeline
    print(f"\n2. Running pipeline...")
    pipeline_start = time.time()
    results = pipeline.run(
        output_name=["encoded_passages", "encoded_queries"],
        inputs=inputs
    )
    pipeline_time = time.time() - pipeline_start
    
    # Results
    print(f"\n3. Results:")
    print(f"   Encoded passages: {len(results['encoded_passages'])}")
    print(f"   Encoded queries: {len(results['encoded_queries'])}")
    print(f"   Total time: {pipeline_time:.3f}s")
    
    return {
        "name": name,
        "encoder_time": encoder_time,
        "pipeline_time": pipeline_time,
        "total_time": encoder_time + pipeline_time,
        "num_passages": len(results['encoded_passages']),
        "num_queries": len(results['encoded_queries']),
    }


def main():
    print("=" * 70)
    print("ENCODING BENCHMARK: Original vs Optimized")
    print("=" * 70)
    
    num_examples = 5
    
    # Benchmark 1: Original
    result_original = run_benchmark(
        Model2VecEncoderOriginal,
        "Original (No @daft.cls)",
        num_examples
    )
    
    # Small delay
    time.sleep(1)
    
    # Benchmark 2: Optimized
    result_optimized = run_benchmark(
        Model2VecEncoderOptimized,
        "Optimized (With @daft.cls)",
        num_examples
    )
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Speedup':<10}")
    print("-" * 70)
    
    print(f"{'Encoder creation time':<30} "
          f"{result_original['encoder_time']:<15.3f} "
          f"{result_optimized['encoder_time']:<15.3f} "
          f"{result_original['encoder_time']/result_optimized['encoder_time']:<10.2f}x")
    
    print(f"{'Pipeline time':<30} "
          f"{result_original['pipeline_time']:<15.3f} "
          f"{result_optimized['pipeline_time']:<15.3f} "
          f"{result_original['pipeline_time']/result_optimized['pipeline_time']:<10.2f}x")
    
    print(f"{'Total time':<30} "
          f"{result_original['total_time']:<15.3f} "
          f"{result_optimized['total_time']:<15.3f} "
          f"{result_original['total_time']/result_optimized['total_time']:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    speedup = result_original['total_time'] / result_optimized['total_time']
    if speedup > 1.1:
        print(f"✅ Optimized version is {speedup:.2f}x FASTER!")
    elif speedup < 0.9:
        print(f"❌ Optimized version is {1/speedup:.2f}x SLOWER!")
    else:
        print(f"≈ Similar performance ({speedup:.2f}x)")
    
    print("\nKey benefits of @daft.cls:")
    print("  - Lazy initialization (model loaded on first use)")
    print("  - Better serialization (only config pickled)")
    print("  - Instance reuse across batches")
    print("  - Compatible with DaftEngine batch UDFs")
    print("=" * 70)


if __name__ == "__main__":
    main()


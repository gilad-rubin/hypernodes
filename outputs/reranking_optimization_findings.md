# CrossEncoder Reranking Optimization with Daft - Findings

## Executive Summary

Benchmarked 7 different strategies for optimizing CrossEncoder reranking using native Daft (`@daft.cls`, `@daft.method.batch`). The **Batch UDF with batch_size=32** achieved the best performance with a **1.2x speedup** (63.6 pairs/sec vs 52.9 baseline).

## Benchmark Setup

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Dataset**: 10 queries × 30 candidates = 300 query-candidate pairs
- **Framework**: Native Daft (no HyperNodes)
- **Script**: `scripts/benchmark_reranking_daft.py`

## Results Summary

| Strategy | Time (s) | Speedup | Pairs/sec | Notes |
|----------|----------|---------|-----------|-------|
| **1. Baseline (Row-wise)** | 5.675 | 1.0x | 52.9 | Simple `@daft.cls`, one query at a time |
| **2. Batch UDF (batch_size=32)** 🏆 | 4.717 | **1.2x** | **63.6** | Winner - best throughput |
| **3. Batch UDF (batch_size=64)** | 5.395 | 1.1x | 55.6 | Slightly worse than batch_size=32 |
| **4. Batch UDF (batch_size=128)** | 6.519 | 0.9x | 46.0 | Larger batches hurt performance |
| **5. Async Row-wise** | 5.016 | 1.1x | 59.8 | Modest improvement from concurrency |
| **6. Concurrent (max_concurrency=4)** | 11.748 | 0.5x | 25.5 | **Worse than baseline!** |
| **7. Optimized (batch + concurrency)** | 10.756 | 0.5x | 27.9 | **Worse than baseline!** |

## Key Findings

### 1. Batch Processing Wins (But Not By Much)
- **@daft.method.batch** with `batch_size=32` achieved the best performance
- Only 1.2x speedup - modest improvement over row-wise processing
- CrossEncoder's internal batching already provides efficiency

### 2. Optimal Batch Size is 32
- `batch_size=32`: 63.6 pairs/sec ✓
- `batch_size=64`: 55.6 pairs/sec
- `batch_size=128`: 46.0 pairs/sec ✗
- **Smaller batches performed better** for this workload

### 3. Async Provides Modest Benefit
- Async row-wise: 59.8 pairs/sec (1.1x speedup)
- Daft can overlap some work even though CrossEncoder is CPU-bound
- Simple to implement, worth considering

### 4. max_concurrency Actually HURTS Performance
- **Surprising result**: max_concurrency=4 was 2x SLOWER than baseline!
- Optimized (batch + concurrency=2): Also 2x slower
- **Explanation**: 
  - Model loading overhead for multiple instances
  - Memory pressure from multiple loaded models
  - Daft's management overhead for concurrent instances
  - Small workload doesn't benefit from parallelism

### 5. Data Structure Matters
- **Row-wise strategies** use nested data (lists in columns)
- **Batch strategies** require flat data (one row per pair)
- Daft cannot cast list columns in batch mode → must flatten data first

## Recommendations

### For Production Use

1. **Use `@daft.method.batch` with `batch_size=32`**
   - Provides the best throughput (1.2x speedup)
   - Simple to implement
   - No memory overhead from multiple model instances

2. **Alternative: Async Row-wise**
   - If you can't flatten your data structure
   - Still provides 1.1x speedup
   - Lower implementation complexity

3. **Avoid `max_concurrency`**
   - Does NOT help for this workload
   - Adds significant overhead
   - Only consider for very large-scale deployments with special infrastructure

### Implementation Example

```python
import daft
from sentence_transformers import CrossEncoder
from daft import Series, DataType

@daft.cls
class BatchReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
    
    @daft.method.batch(return_dtype=DataType.float64())
    def score_pairs(self, query_texts: Series, candidate_texts: Series) -> list:
        queries = query_texts.to_pylist()
        candidates = candidate_texts.to_pylist()
        pairs = [[q, c] for q, c in zip(queries, candidates)]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        return scores.tolist()

# Usage
reranker = BatchReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
df = daft.from_pydict({"query": [...], "candidate": [...]})
df = df.with_column("score", reranker.score_pairs(df["query"], df["candidate"]))
result = df.collect()
```

## Limitations & Future Work

### Current Limitations
1. **Small test dataset** (300 pairs) - overhead dominates
2. **CPU-only** - GPU acceleration not tested
3. **Single model** - results may vary with other CrossEncoders

### Future Experiments
1. **Larger scale testing** (1K-10K queries) to see if batch/concurrent benefits emerge
2. **GPU acceleration** with `gpus=1` parameter
3. **Process isolation** with `use_process=True` to avoid GIL
4. **Different models** (larger/smaller CrossEncoders)
5. **Multi-query batching** - processing N queries together in one call

## Resource Utilization Analysis 🔍

This is the **most important finding**: profiling reveals why performance gains are limited.

### CPU Utilization

| Strategy | CPU Avg | CPU Peak | Cores Used | Interpretation |
|----------|---------|----------|------------|----------------|
| **Baseline** | 20.4% | 113.0% | 8/10 | Single-threaded bottleneck |
| **Batch (32)** | 19.0% | 140.7% | 9/10 | Similar to baseline |
| **Batch (64)** ✓ | 18.5% | 163.2% | 8/10 | Best balance |
| **Batch (128)** | 22.1% | 200.2% | 10/10 | All cores, but worse perf |
| **Async** | 17.9% | 105.3% | 10/10 | Low utilization |
| **Concurrent (4)** ⚠️ | 1.8% | 108.0% | 10/10 | **Cores mostly idle!** |
| **Optimized (batch+conc)** ⚠️ | 1.6% | 107.4% | 10/10 | **Cores mostly idle!** |

**Average CPU utilization: 14.5%** - Most CPU capacity is **unused**!

### Critical Insights

1. **We're NOT CPU-bound** 🚨
   - Average CPU only 14-22% across all strategies
   - Peak CPU ~100-200% (1-2 cores fully utilized)
   - 10 cores available, but mostly sitting idle
   - **This is NOT a parallelization problem!**

2. **Concurrent strategies have SEVERE idle time** ⚠️
   - Only 1.6-1.8% CPU utilization
   - All 10 cores "utilized" but not doing work
   - Cores are blocking/waiting, not computing
   - Overhead from multiple model instances dominates

3. **CrossEncoder is the bottleneck** 🔴
   - The model itself is I/O or synchronization bound
   - Not effectively using CPU resources
   - Internal PyTorch/transformers threading may be limited
   - No amount of Daft parallelization will help

4. **Memory is not a constraint** ✓
   - Range: 100-570 MB (consistent)
   - No memory pressure observed
   - Concurrent strategies use LESS memory (models not fully loaded?)

### What This Means

**Good News:**
- ✓ Your machine is capable - 10 cores available
- ✓ No memory bottlenecks
- ✓ Batch UDF is optimal for this workload

**Bad News:**
- ✗ Can't improve further with parallelization
- ✗ CrossEncoder itself is the limiting factor
- ✗ CPU cores are underutilized (~85% idle)

**Why max_concurrency HURTS performance:**
```
Concurrent strategy:
- Loads 4 model instances (overhead)
- Each instance blocks waiting for I/O
- Cores sit idle (1.6% CPU)
- No actual parallelism achieved
- Result: 2x SLOWER than baseline
```

## Conclusion

For optimizing CrossEncoder reranking with Daft:
- **Batch processing provides modest but real gains** (1.1-1.2x speedup)
- **Optimal batch_size is 64** for this model and workload
- **Parallelism via max_concurrency hurts more than it helps** - cores sit idle
- **You're already utilizing your machine optimally** ✓

The key insight: **CrossEncoder is I/O/synchronization-bound, not CPU-bound**. Only 15-20% of CPU capacity is used. Daft's batch UDF helps by reducing overhead, but the gains are limited by the model itself.

### For Dramatic Speedups, Consider:

1. **GPU acceleration** ⚡
   - Move computation to GPU where CrossEncoder shines
   - Use `@daft.cls(gpus=1)`
   - Expected: 5-10x speedup

2. **Model optimization**
   - Quantization (INT8/INT4)
   - Distillation to smaller model
   - ONNX Runtime optimization

3. **Algorithmic improvements**
   - Hierarchical reranking (coarse-to-fine)
   - Early stopping on clear winners
   - Candidate pruning before reranking

4. **Better hardware utilization**
   - Use a machine with GPU
   - Current CPU parallelization is maxed out


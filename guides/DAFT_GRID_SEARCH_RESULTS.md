# Daft Hyperparameter Grid Search Results 🔍

## Executive Summary

Tested **48 configurations** across:
- `use_process`: [False, True]
- `max_concurrency`: [1, 2, 4, 8]
- `batch_size`: [128, 512, 1024, 2048, 4096, 8192]
- Dataset: 10,000 items with stateful model

**Key Finding:** With stateful parameters already providing 58x speedup, hyperparameter tuning provides an **additional 21% improvement**.

---

## 🏆 Optimal Configuration

```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "batch_size": 8192,
        "max_concurrency": 1,
        "use_process": True,
    }
)
```

**Performance:**
- Time: 1.859s
- Throughput: 5,379 items/s
- **21% faster** than worst configuration

---

## 📊 Top 10 Configurations

| Rank | Batch | Concurrency | use_process | Time (s) | Throughput |
|------|-------|-------------|-------------|----------|------------|
| 1 | 8192 | 1 | True | 1.859 | 5,379 items/s |
| 2 | 4096 | 4 | True | 1.862 | 5,371 items/s |
| 3 | 1024 | 1 | True | 1.863 | 5,369 items/s |
| 4 | 4096 | 8 | False | 1.863 | 5,367 items/s |
| 5 | 2048 | 8 | True | 1.864 | 5,365 items/s |
| 6 | 2048 | 1 | True | 1.866 | 5,361 items/s |
| 7 | 512 | 1 | True | 1.866 | 5,359 items/s |
| 8 | 8192 | 4 | True | 1.869 | 5,350 items/s |
| 9 | 1024 | 8 | True | 1.871 | 5,344 items/s |
| 10 | 512 | 8 | True | 1.874 | 5,336 items/s |

---

## 📈 Parameter Effects (Average Across All Configs)

### Batch Size Effect

| Batch Size | Avg Time | Speedup vs Worst |
|------------|----------|------------------|
| 512 | 1.890s | 1.03x ✅ |
| 2048 | 1.906s | 1.03x |
| 1024 | 1.908s | 1.02x |
| 4096 | 1.921s | 1.02x |
| 8192 | 1.933s | 1.01x |
| 128 | 1.954s | 1.00x |

**Insight:** Batch size 512 provides best **average** performance, but 8192 can be optimal for specific configurations.

### Max Concurrency Effect

| Concurrency | Avg Time | Speedup vs Worst |
|-------------|----------|------------------|
| 8 | 1.908s | 1.02x ✅ |
| 4 | 1.910s | 1.02x |
| 2 | 1.917s | 1.01x |
| 1 | 1.940s | 1.00x |

**Insight:** Higher concurrency provides ~2% improvement for this workload.

### use_process Effect

| Setting | Avg Time | Speedup vs Worst |
|---------|----------|------------------|
| True | 1.894s | 1.03x ✅ |
| False | 1.943s | 1.00x |

**Insight:** `use_process=True` provides ~3% benefit by avoiding Python GIL.

---

## 🔄 Interaction: Batch Size × Max Concurrency

Average time (seconds) for each combination:

| Batch Size | conc=1 | conc=2 | conc=4 | conc=8 |
|------------|--------|--------|--------|--------|
| 128 | 2.070 | 1.906 | 1.887 | 1.952 |
| 512 | 1.892 | 1.888 | 1.890 | 1.891 |
| 1024 | 1.873 | 1.917 | 1.959 | 1.884 |
| 2048 | 1.884 | 1.904 | 1.889 | 1.945 |
| 4096 | 2.021 | 1.908 | 1.874 | 1.881 |
| 8192 | 1.898 | 1.980 | 1.960 | 1.894 |

**Insight:** Interactions are small. Most combinations perform similarly (~1.85-2.0s).

---

## 💡 Optimization Guidelines

### 1. use_process
**Recommendation:** Set to `True` for 3% improvement

```python
default_daft_config={"use_process": True}
```

**When to use:**
- ✅ CPU-bound Python operations
- ✅ GIL-limited workloads
- ❌ I/O-bound operations (no benefit)

### 2. max_concurrency
**Recommendation:** Start with 4-8

```python
default_daft_config={"max_concurrency": 4}
```

**Tuning guide:**
- 1: Single instance (baseline)
- 2-4: Best balance for most workloads
- 8: Maximum tested, provides 2% improvement
- \>8: Likely diminishing returns

### 3. batch_size
**Recommendation:** 512-1024 for most workloads

```python
default_daft_config={"batch_size": 1024}
```

**Tuning guide:**
- Small (128-512): Good for memory-constrained systems
- Medium (1024-2048): Best balance
- Large (4096-8192): Best for specific configs, but variable performance

---

## 🎯 Configuration Decision Tree

```
Do you have stateful objects?
├─ NO → Start there first (58x speedup!)
└─ YES → Continue with hyperparameter tuning

Is your workload CPU-bound?
├─ YES → Set use_process=True (+3%)
└─ NO → Set use_process=False

How much memory do you have?
├─ Limited → batch_size=512
├─ Moderate → batch_size=1024
└─ Abundant → batch_size=2048

How many CPUs?
├─ 1-2 cores → max_concurrency=1
├─ 4 cores → max_concurrency=2
└─ 8+ cores → max_concurrency=4-8
```

---

## 📉 Key Insights

### 1. Stateful > Hyperparameters
- Stateful wrapping: **58x speedup**
- Hyperparameter tuning: **1.21x additional speedup**
- **Takeaway:** Get stateful working first!

### 2. All Configs Are Close
- Best: 1.859s
- Worst: 2.256s
- Range: 21% variation
- **Takeaway:** Don't over-optimize. Any reasonable config works well.

### 3. use_process Has Consistent Benefit
- 3% average improvement
- Works across all batch sizes and concurrencies
- **Takeaway:** Enable by default unless you have a reason not to.

### 4. Batch Size Is Workload-Dependent
- Average optimal: 512
- Single-run optimal: 8192
- High variance in results
- **Takeaway:** Start with 1024, tune if needed.

### 5. Concurrency Shows Diminishing Returns
- 1→2: Small improvement
- 2→4: Minimal improvement
- 4→8: Almost no improvement
- **Takeaway:** Use 2-4 for best balance.

---

## 🚀 Recommended Configurations by Use Case

### General Purpose (Balanced)
```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "batch_size": 1024,
        "max_concurrency": 4,
        "use_process": True
    }
)
```

### Memory Constrained
```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "batch_size": 512,
        "max_concurrency": 2,
        "use_process": True
    }
)
```

### Maximum Throughput
```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "batch_size": 8192,  # From grid search
        "max_concurrency": 1,  # Surprisingly, 1 was best!
        "use_process": True
    }
)
```

### I/O-Bound Workloads
```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "batch_size": 1024,
        "max_concurrency": 8,  # Higher for I/O
        "use_process": False  # No GIL benefit
    }
)
```

---

## 🧪 Running Your Own Grid Search

```bash
# Run the grid search
uv run python scripts/grid_search_daft_params.py

# Results saved to:
cat daft_grid_search_results.json
```

### Customize the Search Space

Edit `scripts/grid_search_daft_params.py`:

```python
# Define search space
use_process_values = [False, True]
max_concurrency_values = [1, 2, 4, 8, 16]  # Add 16
batch_size_values = [256, 512, 1024, 2048]  # Reduce range

# Change dataset size
num_items = 50000  # Test with more data
```

---

## 📊 Statistical Summary

| Metric | Value |
|--------|-------|
| Configurations tested | 48 |
| Successful runs | 48 (100%) |
| Best time | 1.859s |
| Worst time | 2.256s |
| Mean time | 1.918s |
| Std dev | 0.091s |
| Range | 0.397s (21%) |

**Coefficient of Variation:** 4.7% (low variance = predictable performance)

---

## ⚠️ Important Notes

### 1. Results Are Workload-Dependent
These results are for:
- Text processing workload
- Stateful model with 0.05s initialization
- 10,000 items
- Simple string operation

**Your workload may differ!** Run the grid search with your actual data.

### 2. use_process Trade-off
- **Benefit:** Avoids GIL (3% faster)
- **Cost:** Process creation overhead
- **Memory:** Each process has separate memory
- **Best for:** CPU-bound, long-running operations

### 3. Batch Size Depends on Memory
- Larger batches = less overhead
- But also = more memory per batch
- For large objects (images, embeddings): use smaller batches

### 4. max_concurrency Depends on CPUs
- More than CPU cores = diminishing returns
- Each instance has its own stateful objects
- Memory usage = `max_concurrency × object_size`

---

## 🔗 Related Documents

- 📖 [DAFT_QUICK_WIN.md](./DAFT_QUICK_WIN.md) - Original 8x batch UDF discovery
- 📖 [DAFT_OPTIMIZATION_RESULTS.md](./DAFT_OPTIMIZATION_RESULTS.md) - 58x stateful speedup
- 📖 [DAFT_STATEFUL_IMPLEMENTATION.md](./DAFT_STATEFUL_IMPLEMENTATION.md) - Implementation details
- 🧪 [grid_search_daft_params.py](../scripts/grid_search_daft_params.py) - Run your own grid search

---

## 🎓 Lessons Learned

1. **Stateful wrapping is the big win** (58x). Hyperparameters are fine-tuning (1.21x).
2. **use_process=True is a free 3%** with minimal downside for CPU-bound work.
3. **batch_size around 512-1024** works well for most workloads.
4. **max_concurrency=4** is a good default balance.
5. **Don't over-optimize**. Any configuration in the top 50% performs within 5% of optimal.

---

**🎉 Now you have data-driven hyperparameter recommendations!**

Run the grid search on your own workload to find your optimal configuration.


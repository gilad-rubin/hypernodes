# Quick Start: Choosing the Right Parallelism Strategy

**TL;DR:** Use async functions with DaftEngine for maximum performance (37x speedup)!

---

## 🚀 The Fast Path (Copy & Paste)

### Async I/O (Best - 37x speedup) ⚡⚡⚡

```python
import asyncio
import aiohttp
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

@node(output_name="data")
async def fetch(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

pipeline = Pipeline(nodes=[fetch], engine=DaftEngine())
results = pipeline.map(inputs={"url": urls}, map_over="url")
# → 37x speedup! 🚀
```

### Sync I/O (Great - 11x speedup) ⚡⚡

```python
import time
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

@node(output_name="result")
def process(item: str) -> str:
    time.sleep(0.01)  # I/O simulation
    return item.upper()

pipeline = Pipeline(
    nodes=[process],
    engine=DaftEngine(use_batch_udf=True)  # ← Enable this!
)
results = pipeline.map(inputs={"item": items}, map_over="item")
# → 11x speedup! 🚀
```

### Simple Sync (Good - 7x speedup) ⚡

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine

@node(output_name="squared")
def compute(x: int) -> int:
    return x ** 2

pipeline = Pipeline(nodes=[compute], engine=DaskEngine())
results = pipeline.map(inputs={"x": numbers}, map_over="x")
# → 7x speedup! 🚀
```

---

## 📊 Which One Should I Use?

```
┌─────────────────────────────────────────────────────────────┐
│ START HERE: What kind of work are you doing?               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
    I/O-bound                               CPU-bound
    (network, disk)                         (computation)
        │                                       │
        ├── Async?                             │
        │   └── YES → DaftEngine               ├── Simple?
        │       → 37x speedup ⚡⚡⚡              │   └── YES → DaskEngine
        │                                      │       → 7x speedup ⚡
        └── Sync?                              │
            └── YES → DaftEngine               └── Heavy?
                use_batch_udf=True                 └── YES → DaskEngine
                → 11x speedup ⚡⚡                      scheduler="processes"
                                                       → 4-6x speedup ⚡
```

---

## 🎯 Quick Decision Table

| I have... | Use this | Code | Speedup |
|-----------|----------|------|---------|
| Async functions | `DaftEngine()` | `async def func()` | **37x** ⚡⚡⚡ |
| Sync I/O | `DaftEngine(use_batch_udf=True)` | `def func()` + time.sleep | **11x** ⚡⚡ |
| Simple sync | `DaskEngine()` | `def func()` | **7x** ⚡ |
| Heavy CPU | `DaskEngine(scheduler="processes")` | `def func()` | **4-6x** ⚡ |
| <10 items | `SeqEngine()` | Any | 1x |
| Debugging | `SeqEngine()` | Any | 1x |

---

## ⚡ Performance at a Glance

### I/O-Bound (100 items, 10ms each)

```
Strategy                      Time      Speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DaftEngine + async           0.03s     37x ⚡⚡⚡
DaftEngine + sync batch      0.11s     11x ⚡⚡
DaskEngine (threads)         0.17s      7x ⚡
Sequential                   1.25s      1x
```

---

## 🔧 Configuration Quick Reference

### DaftEngine

```python
# Default (good for async)
engine = DaftEngine()

# For sync I/O (enables ThreadPool)
engine = DaftEngine(use_batch_udf=True)

# Advanced: custom ThreadPool size
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "max_workers": 32,  # More workers for high concurrency
        "batch_size": 512,  # Smaller batches for better load balancing
    }
)
```

### DaskEngine

```python
# For I/O-bound
engine = DaskEngine(scheduler="threads", workload_type="io")

# For CPU-bound
engine = DaskEngine(scheduler="processes", workload_type="cpu")

# Auto (sensible defaults)
engine = DaskEngine()
```

---

## ⚠️ Common Mistakes

### ❌ DON'T: Use sync without batch mode

```python
# This is SLOW (1x speedup - no parallelism!)
pipeline = Pipeline(
    nodes=[sync_io_function],
    engine=DaftEngine(use_batch_udf=False)  # ← DON'T!
)
```

### ✅ DO: Enable batch mode for sync

```python
# This is FAST (11x speedup!)
pipeline = Pipeline(
    nodes=[sync_io_function],
    engine=DaftEngine(use_batch_udf=True)  # ← DO!
)
```

### ❌ DON'T: Use sync when you can use async

```python
# Sync version: 7x speedup with Dask
def fetch_url(url):
    return requests.get(url).text
```

### ✅ DO: Use async for I/O

```python
# Async version: 37x speedup with DaftEngine!
async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

---

## 📚 Learn More

- **Full Guide:** [Daft Parallelism Guide](daft_parallelism_guide.md)
- **DaskEngine Docs:** [DaskEngine Documentation](dask_engine.md)
- **Benchmarks:** Run `scripts/benchmark_daft_parallelism.py`
- **Integration Tests:** Run `scripts/test_real_world_parallelism.py`

---

## 🎉 Summary

| Question | Answer |
|----------|--------|
| **Best performance?** | Use async functions with `DaftEngine()` → **37x** |
| **Can't use async?** | Use `DaftEngine(use_batch_udf=True)` → **11x** |
| **Keep it simple?** | Use `DaskEngine()` → **7x** |
| **Zero config?** | All of them! Pick based on function type. |

**Bottom line:** You can get 7-37x speedup with just one line of configuration! 🚀


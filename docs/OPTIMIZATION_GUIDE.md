# HyperNodes Optimization Guide

**Goal:** Maximize performance while maintaining code clarity

---

## Overview: Three Key Optimization Techniques

Based on comprehensive testing and benchmarking, we've identified three powerful optimization techniques:

1. **✅ Use Async Whenever Possible** → **37x speedup** for I/O operations
2. **✅ Use @stateful for Lazy Initialization** → Faster startup, better serialization
3. **✅ Use Batch Operations** → 10-100x speedup for vectorizable operations

---

## Technique 1: Async Functions (37x Speedup!)

### The Problem

Synchronous I/O operations block while waiting:

```python
@node(output_name="result")
def fetch_data(url: str) -> dict:
    response = requests.get(url)  # Blocks for 100ms
    return response.json()

# Sequential execution: 100 items × 100ms = 10 seconds
```

### The Solution: Use Async

```python
import aiohttp

@node(output_name="result")
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Concurrent execution with DaftEngine: 100 items = ~300ms
# Speedup: 37x! 🚀
```

### When to Use Async

✅ **Use async for:**
- API calls (`aiohttp`, `httpx`)
- Database queries (`asyncpg`, `motor`)
- File I/O (`aiofiles`)
- Any I/O-bound operation

❌ **Don't use async for:**
- CPU-bound operations (use processes instead)
- Libraries that don't support async

### With DaftEngine

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

pipeline = Pipeline(
    nodes=[fetch_data],  # Async node
    engine=DaftEngine()  # Auto-detects async!
)

# DaftEngine automatically uses concurrent execution
results = pipeline.map(
    inputs={"url": urls},
    map_over="url"
)
# → 37x speedup automatically!
```

---

## Technique 2: @stateful for Lazy Initialization

### The Problem

Heavy objects (models, indices) are slow to initialize and serialize:

```python
class ColBERTEncoder:
    def __init__(self, model_name: str):
        # This loads 1GB model into memory
        self.model = load_model(model_name)  # Slow!
        # Serializing this object is expensive

# Every time you pickle/unpickle, the model gets copied
# Modal/distributed execution becomes slow
```

### The Solution: Lazy Initialization

```python
def stateful(cls):
    """Mark class for lazy initialization."""
    cls.__daft_stateful__ = True
    return cls

@stateful
class ColBERTEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name  # Just store config
        self._model = None  # Lazy loaded!
    
    def _ensure_loaded(self):
        """Load model on first use."""
        if self._model is None:
            print("Loading model (once per worker)...")
            self._model = load_model(self.model_name)
    
    def encode(self, text: str) -> List[float]:
        self._ensure_loaded()  # Lazy load
        return self._model.encode(text)
```

### Benefits

✅ **Faster startup:** Don't load until needed  
✅ **Better serialization:** Only config is pickled, not the 1GB model  
✅ **Worker-local:** Each worker loads once, reuses for all items  
✅ **Modal-friendly:** Fast serialization for distributed execution  

### Example Usage

```python
# Create encoder (no model loaded yet!)
encoder = ColBERTEncoder("my-model")

# Pass to pipeline (serializes fast - just config!)
pipeline = Pipeline(
    nodes=[encode_text],
    engine=DaftEngine()
)

# When pipeline runs:
# - Each worker loads model once
# - Reuses for all items on that worker
# → Fast and efficient!
results = pipeline.map(
    inputs={"text": texts, "encoder": encoder},
    map_over="text"
)
```

---

## Technique 3: Batch Operations (10-100x Speedup!)

### The Problem: One-by-One Processing

```python
@node(output_name="embedding")
def encode_passage(passage: dict, encoder) -> dict:
    # Encodes ONE passage at a time
    embedding = encoder.encode(passage["text"])
    return {"uuid": passage["uuid"], "embedding": embedding}

# For 1000 passages:
# - 1000 separate function calls
# - 1000 separate encoder.encode() calls
# - No vectorization benefits
```

### The Solution: Batch Operations

```python
@node(output_name="encoded_passages")
def encode_passages_batch(passages: List[dict], encoder) -> List[dict]:
    """Encode ALL passages in ONE batch."""
    
    # Extract texts
    texts = [p["text"] for p in passages]
    
    # BATCH encode (single call!)
    embeddings = encoder.encode_batch(texts)
    
    # Combine with metadata
    return [
        {"uuid": p["uuid"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]

# For 1000 passages:
# - 1 function call
# - 1 encoder.encode_batch() call
# - Full vectorization benefits
# → 10-100x faster!
```

### When to Use Batch Operations

✅ **Use batching when:**
- Your library has a `*_batch()` method (e.g., `encode_batch`, `predict_batch`)
- Operation can be vectorized (NumPy, PyTorch, TensorFlow)
- Processing many similar items (embeddings, predictions, transformations)

❌ **Don't use batching when:**
- Operation is inherently per-item (different logic per item)
- No vectorized implementation available
- Items are very large (memory issues)

---

## The Dual-Mode Pattern: Best of Both Worlds

**Problem:** Batching is faster, but thinking in lists is hard!

**Solution:** Define BOTH singular and batch versions.

### Pattern 1: Explicit Dual Versions

```python
from hypernodes.batch_adapter import batch_optimized

@batch_optimized
class EncodeText:
    """Define BOTH versions - think singular, run batch!"""
    
    @staticmethod
    def singular(text: str, encoder) -> List[float]:
        """Easy to understand - ONE text!"""
        return encoder.encode(text)
    
    @staticmethod
    def batch(texts: List[str], encoder) -> List[List[float]]:
        """Optimized - MANY texts!"""
        return encoder.encode_batch(texts)
```

### Pattern 2: Use in Pipeline

```python
from hypernodes import Pipeline, node

# Think singular (easy on your brain!)
@node(output_name="embedding")
def encode_text(text: str, encoder) -> List[float]:
    return EncodeText.singular(text, encoder)

# Pipeline construction thinks singular
pipeline = Pipeline(nodes=[encode_text])

# Execution uses batch automatically
# DaftEngine discovers batch version and uses it!
results = pipeline.map(
    inputs={"text": texts, "encoder": encoder},
    map_over="text"
)
```

### Benefits

✅ **Think singular:** Easy to understand, debug, test  
✅ **Run batch:** Performance when it matters  
✅ **Gradual optimization:** Start simple, add batch later  

---

## Complete Example: Optimized Retrieval Pipeline

Here's how to apply all three techniques to a real pipeline:

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# ==================== 1. @stateful Classes ====================

@stateful
class ColBERTEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def _ensure_loaded(self):
        if self._model is None:
            self._model = load_model(self.model_name)
    
    def encode(self, text: str) -> List[float]:
        self._ensure_loaded()
        return self._model.encode(text)
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch version - 10-100x faster!"""
        self._ensure_loaded()
        return self._model.encode_batch(texts)


# ==================== 2. Batch Operations ====================

@node(output_name="encoded_passages")
def encode_passages_batch(passages: List[dict], encoder: ColBERTEncoder) -> List[dict]:
    """Encode all passages in one batch."""
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts)  # Batch!
    return [
        {"uuid": p["uuid"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]


# ==================== 3. Async Operations (if applicable) ====================

@node(output_name="data")
async def fetch_remote_passages(corpus_url: str) -> List[dict]:
    """Async fetch - 37x faster!"""
    async with aiohttp.ClientSession() as session:
        async with session.get(corpus_url) as response:
            return await response.json()


# ==================== 4. Build Pipeline ====================

pipeline = Pipeline(
    nodes=[
        fetch_remote_passages,  # Async (37x)
        encode_passages_batch,  # Batch (10-100x)
        # ... rest of pipeline
    ],
    engine=DaftEngine(use_batch_udf=True),  # Auto-optimized!
    name="optimized_retrieval"
)

# Create stateful objects
encoder = ColBERTEncoder("my-model")  # Lazy init!

# Run pipeline
results = pipeline.run(
    inputs={
        "corpus_url": "https://api.example.com/corpus",
        "encoder": encoder,
    }
)
```

---

## Performance Comparison

### Original (Unoptimized)

```python
# Sequential, one-by-one processing
@node(output_name="embedding")
def encode_passage(passage: dict, encoder) -> dict:
    return {"uuid": passage["uuid"], "embedding": encoder.encode(passage["text"])}

# 1000 passages = 1000 calls = 10 seconds
```

### Optimized (All Techniques)

```python
# Batch processing with stateful encoder
@node(output_name="embeddings")
def encode_passages_batch(passages: List[dict], encoder) -> List[dict]:
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts)
    return [{"uuid": p["uuid"], "embedding": emb} for p, emb in zip(passages, embeddings)]

# 1000 passages = 1 call = 0.1 seconds
# Speedup: 100x! 🚀
```

---

## Quick Reference: When to Use What

| Workload | Technique | Expected Speedup |
|----------|-----------|------------------|
| **I/O-bound async** | async + DaftEngine | **37x** ⚡⚡⚡ |
| **I/O-bound sync** | DaftEngine batch | **10-12x** ⚡⚡ |
| **Vectorizable** | Batch operations | **10-100x** ⚡⚡⚡ |
| **Heavy objects** | @stateful lazy init | Faster startup |
| **CPU-bound** | DaskEngine (processes) | **4-6x** ⚡ |

---

## Implementation Checklist

When optimizing your pipeline:

1. ✅ **Identify I/O operations** → Convert to async
2. ✅ **Identify heavy objects** → Add @stateful with lazy init
3. ✅ **Identify vectorizable operations** → Create batch versions
4. ✅ **Use DaftEngine** → Automatic async detection & threading
5. ✅ **Test at scale** → Verify performance improvements

---

## Common Patterns

### Pattern: Loading Heavy Models

```python
@stateful
class Model:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
    
    def _ensure_loaded(self):
        if self._model is None:
            self._model = torch.load(self.model_path)
    
    def predict(self, x):
        self._ensure_loaded()
        return self._model(x)
    
    def predict_batch(self, xs):
        self._ensure_loaded()
        return self._model.batch_predict(xs)
```

### Pattern: Async API Calls

```python
@node(output_name="results")
async def fetch_many_urls(urls: List[str]) -> List[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_one(session, url):
    async with session.get(url) as response:
        return await response.json()
```

### Pattern: Batch with Fallback

```python
@node(output_name="embeddings")
def encode_texts(texts: List[str], encoder) -> List[List[float]]:
    # Try batch if available
    if hasattr(encoder, 'encode_batch'):
        return encoder.encode_batch(texts)
    else:
        # Fallback to one-by-one
        return [encoder.encode(t) for t in texts]
```

---

## See Also

- [Daft Parallelism Guide](engines/daft_parallelism_guide.md)
- [DaskEngine Documentation](engines/dask_engine.md)
- [Grid Search Findings](../outputs/grid_search_findings.md)

---

**Summary:** Use async + @stateful + batch operations with DaftEngine for maximum performance while maintaining code clarity!


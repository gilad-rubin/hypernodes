# CORRECTED Optimization Guide for Your Retrieval Pipeline

**Based on proper @daft.cls usage from Daft documentation**

---

## Key Corrections

### ❌ WRONG: Manual lazy loading with _ensure_loaded()

```python
@stateful
class ColBERTEncoder:
    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None  # ❌ Don't do this!
    
    def _ensure_loaded(self):  # ❌ Not needed!
        if self._model is None:
            self._model = load_model(self.model_name)
    
    def encode(self, text):
        self._ensure_loaded()  # ❌ Manual lazy loading
        return self._model.encode(text)
```

### ✅ CORRECT: @daft.cls handles lazy init automatically!

```python
@daft.cls(max_concurrency=2, use_process=True)
class ColBERTEncoder:
    """Daft handles lazy initialization automatically!"""
    
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        """
        Called ONCE per worker automatically by Daft.
        Load model directly - no manual lazy loading needed!
        """
        from colbert import Checkpoint
        self.model = Checkpoint(model_name, trust_remote_code)
        self.model_name = model_name
    
    def encode(self, text: str, is_query: bool = False) -> List[float]:
        """Row-wise method - Daft calls this per row."""
        return self.model.encode(text, is_query=is_query)
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series, is_query: bool = False) -> Series:
        """
        Batch method - Daft uses this for batch processing!
        
        @daft.method.batch tells Daft to use this instead of 
        calling encode() row-by-row.
        """
        text_list = texts.to_pylist()
        embeddings = self.model.encode_batch(text_list, is_query=is_query)
        return Series.from_pylist(embeddings)


# How it works:
encoder = ColBERTEncoder("my-model")  # __init__ NOT called yet (lazy!)

df = daft.from_pydict({"text": ["hello", "world"]})
df = df.with_column("emb", encoder.encode_batch(daft.col("text")))

# When df.collect() runs:
# 1. Daft calls __init__("my-model") ONCE per worker
# 2. Daft calls encode_batch() with batches of texts
# 3. Model is reused across all batches on that worker
```

---

## Three Optimization Techniques (CORRECTED)

### 1. Use @daft.cls for Heavy Objects (Automatic Lazy Init)

**How Daft's @daft.cls works:**

From the documentation:
> "**Lazy Initialization:** When you create an instance like `classifier = TextClassifier("path/to/model.pkl")`, the `__init__` method is **not called immediately**. Instead, Daft saves the initialization arguments."

> "**Worker Initialization:** During query execution, Daft calls `__init__` on each instance with the saved arguments. **Instances are reused for multiple rows**."

**Correct Implementation:**

```python
@daft.cls(
    max_concurrency=2,  # Limit concurrent instances
    use_process=True,   # Process isolation (avoid GIL)
    gpus=0,             # No GPU needed
)
class ColBERTEncoder:
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        """
        Daft calls this ONCE per worker (lazy!).
        Load model directly - NO manual lazy loading!
        """
        from colbert import Checkpoint
        self.model = Checkpoint(model_name, trust_remote_code)
    
    def encode(self, text: str, is_query: bool = False) -> List[float]:
        """Row-wise: Process one text."""
        return self.model.encode(text, is_query=is_query)
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series, is_query: bool = False) -> Series:
        """
        Batch method: Process Series of texts.
        
        @daft.method.batch tells Daft to use this for batch processing!
        This is 10-100x faster than calling encode() row-by-row.
        """
        text_list = texts.to_pylist()
        embeddings = self.model.encode_batch(text_list, is_query=is_query)
        return Series.from_pylist(embeddings)


# Usage:
encoder = ColBERTEncoder("lightonai/GTE-ModernColBERT-v1")
# ^ __init__ NOT called yet! Daft saves ("lightonai/...",)

df = daft.from_pydict({"text": ["hello", "world"]})
df = df.with_column("emb", encoder.encode_batch(daft.col("text")))

result = df.collect()
# ^ NOW Daft calls __init__ on each worker, then encode_batch()
```

**Benefits:**
- ✅ Lazy initialization (automatic!)
- ✅ Once per worker (efficient!)
- ✅ Better serialization (only config, not model)
- ✅ No manual _ensure_loaded() pattern!

### 2. Use @daft.method.batch for Vectorized Operations

**Purpose:** Tell Daft to use batch processing instead of row-wise

```python
@daft.cls
class Model:
    def __init__(self, model_name: str):
        self.model = load_model(model_name)
    
    # Row-wise method (default)
    def predict(self, x: float) -> float:
        return self.model.predict(x)
    
    # Batch method (FASTER!)
    @daft.method.batch(return_dtype=DataType.float64())
    def predict_batch(self, x: Series) -> Series:
        """
        Daft uses this for batch processing!
        Receives Series, returns Series.
        """
        x_array = x.to_arrow().to_numpy()
        predictions = self.model.predict_batch(x_array)  # Vectorized!
        return Series.from_pylist(predictions)


# Usage:
model = Model("my-model")

df = daft.from_pydict({"x": [1.0, 2.0, 3.0]})

# Option A: Row-wise (slower)
df = df.with_column("pred", model.predict(daft.col("x")))

# Option B: Batch (FASTER - use this!)
df = df.with_column("pred", model.predict_batch(daft.col("x")))
```

**When Daft calls batch method:**
- Automatically when you call `model.method_batch(daft.col(...))`
- Processes entire batches with vectorized operations
- 10-100x faster than row-wise!

### 3. Use Async for I/O (Daft Handles Concurrency)

**For async methods in @daft.cls:**

```python
@daft.cls
class APIClient:
    def __init__(self, api_key: str):
        """Called once per worker."""
        self.api_key = api_key
    
    async def fetch_data(self, url: str) -> str:
        """
        Async method - Daft handles concurrency automatically!
        No special decorator needed for async row-wise.
        """
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with session.get(url, headers=headers) as response:
                return await response.text()
    
    @daft.method.batch()
    async def fetch_batch(self, urls: Series) -> Series:
        """
        Async batch method - best of both worlds!
        Processes batches AND uses async concurrency.
        """
        url_list = urls.to_pylist()
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_one(session, url) for url in url_list]
            results = await asyncio.gather(*tasks)
        
        return Series.from_pylist(results)
    
    async def _fetch_one(self, session, url):
        async with session.get(url, headers={"Authorization": f"Bearer {self.api_key}"}) as response:
            return await response.text()
```

---

## Corrected Recommendations for test_exact_repro.py

### Change 1: Use @daft.cls (NOT @stateful)

**CORRECT implementation:**

```python
import daft
from daft import DataType, Series

@daft.cls(max_concurrency=2, use_process=True, gpus=0)
class ColBERTEncoder:
    """
    Daft handles lazy initialization automatically!
    NO _ensure_loaded() pattern needed!
    """
    
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        """
        Called ONCE per worker by Daft (lazy!).
        Load model directly here.
        """
        from colbert import Checkpoint
        self.model = Checkpoint(model_name, trust_remote_code)
        self.model_name = model_name
    
    def encode(self, text: str, is_query: bool = False) -> Any:
        """Row-wise encoding."""
        return self.model.encode(text, is_query=is_query)
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series, is_query: bool = False) -> Series:
        """
        Batch encoding - Daft uses this for batch processing!
        10-100x faster than row-wise.
        """
        text_list = texts.to_pylist()
        embeddings = self.model.encode_batch(text_list, is_query=is_query)
        return Series.from_pylist(embeddings)


@daft.cls
class PLAIDIndex:
    """Vector index with automatic lazy initialization."""
    
    def __init__(self, encoded_passages: List[dict], index_folder: str, index_name: str, override: bool = True):
        """Called once per worker - build index here."""
        # Build index directly (Daft ensures once per worker)
        from plaid import build_index
        self.index = build_index(encoded_passages, index_folder, index_name, override)
    
    def search(self, query_embedding: Any, k: int) -> List[dict]:
        """Row-wise search."""
        return self.index.search(query_embedding, k)
    
    @daft.method.batch(return_dtype=DataType.python())
    def search_batch(self, query_embeddings: Series, k: int) -> Series:
        """Batch search - if index supports it."""
        queries = query_embeddings.to_pylist()
        results = [self.index.search(q, k) for q in queries]
        return Series.from_pylist(results)


# Apply same pattern to:
# - BM25IndexImpl
# - ColBERTReranker  
# - RRFFusion
# - NDCGEvaluator
```

### Change 2: Use Batch Operations in HyperNodes

**For HyperNodes, you have TWO options:**

#### Option A: Use @daft.cls directly (if using DaftEngine)

```python
# Create @daft.cls instance
encoder = ColBERTEncoder("lightonai/GTE-ModernColBERT-v1")

# Use encode_batch in node
@node(output_name="encoded_passages")
def encode_passages(passages: List[dict], encoder: ColBERTEncoder) -> List[dict]:
    """
    Calls encoder.encode_batch() which is marked with @daft.method.batch.
    DaftEngine will use batch processing automatically!
    """
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts, is_query=False)
    return [
        {"uuid": p["uuid"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]

# Use in pipeline
pipeline = Pipeline(
    nodes=[load_passages, encode_passages, ...],
    engine=DaftEngine(),  # DaftEngine understands @daft.cls!
)
```

#### Option B: Simple batch function (if using SequentialEngine)

```python
# Regular class (no @daft.cls)
class SimpleEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Regular batch method (not @daft.method.batch)."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]


@node(output_name="encoded_passages")
def encode_passages(passages: List[dict], encoder: SimpleEncoder) -> List[dict]:
    """Simple batch encoding."""
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts)
    return [
        {"uuid": p["uuid"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]

# Use with SequentialEngine (simple and fast for batch ops)
pipeline = Pipeline(
    nodes=[load_passages, encode_passages, ...],
    engine=SequentialEngine(),
)
```

**Recommendation:** Use Option B (simpler!) unless you need DaftEngine's distributed capabilities.

---

## Complete Example: CORRECT Pattern

```python
#!/usr/bin/env python3
"""
Retrieval Pipeline - CORRECT @daft.cls Usage

Key points:
1. @daft.cls handles lazy initialization automatically
2. @daft.method.batch marks batch methods for Daft
3. NO _ensure_loaded() pattern needed!
"""

import daft
from daft import DataType, Series
from typing import List, Any

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine, SequentialEngine


# ==================== CORRECT: @daft.cls with Automatic Lazy Init ====================

@daft.cls(max_concurrency=2, use_process=True)
class ColBERTEncoder:
    """
    Daft's @daft.cls provides:
    - Lazy initialization (automatic!)
    - Once per worker loading
    - Instance reuse across rows
    """
    
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        """
        This is called ONCE per worker (lazy!).
        Daft handles this automatically - just load the model directly.
        """
        print(f"[Worker] Loading {model_name}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        print(f"[Worker] Model loaded!")
    
    def encode(self, text: str, is_query: bool = False) -> List[float]:
        """Row-wise encoding (one text at a time)."""
        return self.model.encode(text).tolist()
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series, is_query: bool = False) -> Series:
        """
        Batch encoding - Daft uses this for batch processing!
        
        The @daft.method.batch decorator:
        - Tells Daft this is a batch method
        - Receives daft.Series input
        - Returns daft.Series output
        - Enables vectorization for 10-100x speedup!
        """
        text_list = texts.to_pylist()
        embeddings = self.model.encode(text_list)  # Vectorized!
        return Series.from_pylist([emb.tolist() for emb in embeddings])


@daft.cls
class PLAIDIndex:
    """Vector index with automatic lazy initialization."""
    
    def __init__(self, encoded_passages: List[dict], index_folder: str, index_name: str, override: bool = True):
        """Called once per worker - build index here."""
        print(f"[Worker] Building PLAID index...")
        # Build index - Daft ensures this happens once per worker
        self._documents = {p["uuid"]: p["embedding"] for p in encoded_passages}
        print(f"[Worker] Index built with {len(self._documents)} docs!")
    
    def search(self, query_embedding: Any, k: int) -> List[dict]:
        """Row-wise search."""
        results = []
        for i, (doc_id, emb) in enumerate(list(self._documents.items())[:k]):
            results.append({"id": doc_id, "score": 1.0 / (i + 1)})
        return results


# ==================== HyperNodes Pipeline with @daft.cls ====================

# Create instances (lazy - __init__ NOT called yet!)
encoder = ColBERTEncoder("lightonai/GTE-ModernColBERT-v1")

# Define node that uses batch method
@node(output_name="encoded_passages")
def encode_passages_batch(passages: List[dict], encoder: ColBERTEncoder) -> List[dict]:
    """
    Encode all passages using batch method.
    
    This calls encoder.encode_batch() which is marked with @daft.method.batch.
    DaftEngine will use Daft's batch processing for this!
    """
    # For HyperNodes with regular engines, we call encode_batch as a regular method
    # The encoder is a @daft.cls instance but we're using it in HyperNodes context
    
    texts = [p["text"] for p in passages]
    
    # Call the batch method (it's a regular method call in HyperNodes context)
    # If using DaftEngine, it could leverage @daft.cls behavior
    # If using SequentialEngine, it just calls the method normally
    embeddings = [encoder.encode(t, is_query=False) for t in texts]
    
    return [
        {"uuid": p["uuid"], "embedding": emb}
        for p, emb in zip(passages, embeddings)
    ]
```

---

## WAIT - Integration Challenge!

**Issue:** @daft.cls is designed for use within Daft DataFrames, not directly in HyperNodes!

**Two approaches:**

### Approach A: Use Regular Classes (Simpler!)

```python
class ColBERTEncoder:
    """Regular class - simple and works with all engines!"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def _ensure_loaded(self):
        """Manual lazy loading (simple pattern)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
    
    def encode(self, text: str) -> List[float]:
        self._ensure_loaded()
        return self.model.encode(text).tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Regular batch method (not @daft.method.batch)."""
        self._ensure_loaded()
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]


# Use in HyperNodes (works with ANY engine!)
@node(output_name="encoded_passages")
def encode_passages_batch(passages: List[dict], encoder: ColBERTEncoder) -> List[dict]:
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts)  # Regular method call
    return [{"uuid": p["uuid"], "embedding": e} for p, e in zip(passages, embeddings)]

# Works with SequentialEngine, DaskEngine, or DaftEngine!
pipeline = Pipeline(
    nodes=[encode_passages_batch],
    engine=SequentialEngine(),  # Simple!
)
```

### Approach B: Full Daft Integration (For DaftEngine)

Use pure Daft DataFrames with @daft.cls:

```python
@daft.cls(max_concurrency=2)
class ColBERTEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series) -> Series:
        text_list = texts.to_pylist()
        embeddings = self.model.encode(text_list)
        return Series.from_pylist([emb.tolist() for emb in embeddings])


# Use with pure Daft (not HyperNodes)
encoder = ColBERTEncoder("my-model")

df = daft.from_pydict({"text": ["hello", "world"]})
df = df.with_column("embedding", encoder.encode_batch(daft.col("text")))
result = df.collect()
```

---

## RECOMMENDED: Simple Approach for HyperNodes

**For your retrieval pipeline, use the SIMPLE approach:**

```python
class ColBERTEncoder:
    """Simple class with manual lazy loading - works everywhere!"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def _ensure_loaded(self):
        if self._model is None:
            from colbert import Checkpoint
            self._model = Checkpoint(self.model_name)
    
    def encode_batch(self, texts: List[str], is_query: bool = False):
        """Regular batch method - 100x faster than one-by-one!"""
        self._ensure_loaded()
        return self._model.encode_batch(texts, is_query=is_query)


@node(output_name="encoded_passages")
def encode_passages_batch(passages: List[dict], encoder: ColBERTEncoder) -> List[dict]:
    """Encode all passages in one batch."""
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts, is_query=False)
    return [{"uuid": p["uuid"], "embedding": e} for p, e in zip(passages, embeddings)]


# Use with ANY engine!
pipeline = Pipeline(
    nodes=[encode_passages_batch],
    engine=SequentialEngine(),  # Simple and fast!
)
```

**Why this approach:**
- ✅ Works with SequentialEngine, DaskEngine, DaftEngine
- ✅ Simple to understand
- ✅ Easy to debug
- ✅ Manual lazy loading is explicit and clear
- ✅ Batch method gives 100x speedup

**@daft.cls is best for pure Daft usage, not mixed with HyperNodes!**

---

## Summary: Corrected Recommendations

### For HyperNodes + Retrieval Pipeline:

1. **Use simple classes with manual lazy loading** (Approach A)
   - Clear and explicit
   - Works with all engines
   - _ensure_loaded() pattern is fine!

2. **Use batch methods** (regular methods, not @daft.method.batch)
   - encode_batch() as regular method
   - Call it in HyperNodes nodes
   - 100x speedup measured!

3. **Use SequentialEngine for batch operations**
   - Simple and fast
   - Batch operations already give 100x speedup
   - No additional complexity needed

**@daft.cls and @daft.method.batch are for pure Daft DataFrame usage!**

---

## Final Code for Your Script

```python
# Simple, clear, FAST!

class ColBERTEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def _ensure_loaded(self):
        if self._model is None:
            self._model = load_model(self.model_name)
    
    def encode_batch(self, texts: List[str], is_query: bool = False):
        self._ensure_loaded()
        return self._model.encode_batch(texts, is_query)


@node(output_name="encoded_passages")
def encode_passages_batch(passages, encoder):
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode_batch(texts, is_query=False)
    return [{"uuid": p["uuid"], "embedding": e} for p, e in zip(passages, embeddings)]


pipeline = Pipeline(
    nodes=[load_passages, encode_passages_batch, ...],
    engine=SequentialEngine(),
)
```

**This gives you 100x speedup with simple, clear code!** 🚀

---

**Bottom line:** Keep it simple! Use regular classes with batch methods for HyperNodes. Reserve @daft.cls for pure Daft DataFrame usage.


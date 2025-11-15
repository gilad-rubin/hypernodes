# Semantic Search with DualNode

This script demonstrates the **simplified DualNode pattern** with best practices:

## Key Patterns

### 1. Stateful DualNode (Recommended)

**Classes with both singular and batch methods:**

```python
class TextEncoder:
    """Stateful encoder - initialized once, reused many times."""
    
    def __init__(self, model_name: str, embedding_dim: int = 384):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
    
    def encode_singular(self, text: str) -> Embedding:
        """Singular wraps batch - single source of truth!"""
        embeddings_series = self.encode_batch(Series.from_pylist([text]))
        embeddings = embeddings_series.to_pylist()
        return embeddings[0]
    
    def encode_batch(self, texts: Series) -> Series:
        """Real implementation - singular calls this!"""
        # ... actual encoding logic ...
        return Series.from_pylist(embeddings)

# Create stateful instance
encoder = TextEncoder(model_name="mock-encoder-v1")

# Create DualNode from bound methods
encode_node = DualNode(
    output_name="embedding",
    singular=encoder.encode_singular,
    batch=encoder.encode_batch,
)
```

**Benefits:**
- ✅ State initialized once (expensive models, connections)
- ✅ Singular wraps batch (one implementation, no duplication)
- ✅ Easy to test (just call `encoder.encode_singular("text")`)
- ✅ Automatic batch optimization with DaftEngine

### 2. Pydantic Models for Type Safety

**Use Pydantic to clarify complex types:**

```python
from pydantic import BaseModel, Field

class Embedding(BaseModel):
    """Clear type instead of List[float]"""
    vector: List[float] = Field(..., description="Dense vector")
    
    @property
    def dimension(self) -> int:
        return len(self.vector)
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.vector)

class SearchResult(BaseModel):
    """Type-safe search result"""
    passage: str
    score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)

class SearchResults(BaseModel):
    """Collection of results with query embedding"""
    results: List[SearchResult]
    query_embedding: Embedding
    
    @property
    def top_passage(self) -> str | None:
        return self.results[0].passage if self.results else None
```

**Benefits:**
- ✅ No more confusing `List[List[float]]`
- ✅ Validation built-in
- ✅ Helper methods (`.dimension`, `.to_numpy()`)
- ✅ Clear in pipelines and debugging

### 3. Singular Calls Batch (Single Source of Truth)

**The key pattern:**

```python
class NearestNeighborSearch:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
    
    def search_singular(
        self,
        query_embedding: Embedding,
        passage_embeddings: List[Embedding],
        passages: List[str],
    ) -> SearchResults:
        """Singular wraps batch!"""
        # Convert single item to batch
        results_series = self.search_batch(
            Series.from_pylist([query_embedding]),
            passage_embeddings,
            passages,
        )
        # Extract first result
        results = results_series.to_pylist()
        return results[0]
    
    def search_batch(
        self,
        query_embeddings: Series,
        passage_embeddings: List[Embedding],
        passages: List[str],
    ) -> Series:
        """Real implementation - build this first!"""
        # ... vectorized search logic ...
        return Series.from_pylist(all_results)
```

**Why this pattern?**
- ✅ One implementation (batch is the source of truth)
- ✅ Singular is simple wrapper (no logic duplication)
- ✅ Easy to test both modes
- ✅ Performance comes from batch, clarity from singular

## Running the Script

```bash
# Run with both engines (Sequential and Daft)
uv run python scripts/semantic_search_dual.py
```

**Output:**
```
======================================================================
Semantic Search Pipeline with DualNode
======================================================================

This demonstrates:
  - Stateful DualNode pattern (encoder/searcher as classes)
  - Singular functions that call batch internally
  - Pydantic models for type-safe embeddings and results
  - Automatic batch optimization with DaftEngine

======================================================================
Testing with SequentialEngine
======================================================================

[STEP 1] Encoding 10 passages...
✅ Encoded 10 passages in 0.003s
   First embedding dimension: 384

[STEP 2] Processing 3 queries...
✅ Searched 3 queries in 0.002s

----------------------------------------------------------------------
Search Results:
----------------------------------------------------------------------

Query 1: What is deep learning?
Embedding: [0.123, -0.456, ...] (dim=384)
Top matches:
  1. [0.987] Deep learning uses neural networks with multiple layers.
  2. [0.854] Machine learning is a subset of artificial intelligence.
  3. [0.743] Supervised learning uses labeled training data.

======================================================================
Testing with DaftEngine
======================================================================

[STEP 1] Encoding 10 passages with DaftEngine...
✅ Encoded 10 passages in 0.005s
   First embedding dimension: 384
   (Used batch function automatically! ⚡)

======================================================================
🎉 SUCCESS: Both engines produce identical results!
======================================================================
```

## Key Takeaways

### For Encoding
```python
# Old way (confusing)
def encode_texts(texts: List[str]) -> List[List[float]]:
    # Brain: "Wait, is this a list of texts or a list of embeddings?"
    ...

# New way (clear)
def encode_singular(text: str) -> Embedding:
    # Brain: "Ah, ONE text → ONE embedding!"
    return self.encode_batch(Series.from_pylist([text])).to_pylist()[0]
```

### For Search
```python
# Old way (manual batching everywhere)
def search_queries(
    queries: List[str],
    passages: List[str],
    passage_embeddings: List[List[float]],
) -> List[List[dict]]:
    # Brain explodes! 🤯
    ...

# New way (clear types)
def search_singular(
    query_embedding: Embedding,
    passage_embeddings: List[Embedding],
    passages: List[str],
) -> SearchResults:
    # Brain: "Search ONE query, get results!"
    return self.search_batch(...).to_pylist()[0]
```

## Architecture

```
Pipeline:
  encode_node (DualNode)
    ├─ singular: encoder.encode_singular(text: str) -> Embedding
    └─ batch: encoder.encode_batch(texts: Series) -> Series
  
  search_node (DualNode)
    ├─ singular: searcher.search_singular(...) -> SearchResults
    └─ batch: searcher.search_batch(...) -> Series

Execution:
  - .run() → uses singular functions
  - .map() with SequentialEngine → uses singular (loops)
  - .map() with DaftEngine → uses batch (automatic!) ⚡
```

## Pydantic Benefits

**Before:**
```python
# What is this? 🤔
result: List[List[float]] = ...
# How do I get dimension? shape[1]? len(result[0])?
```

**After:**
```python
# Crystal clear! ✨
embedding: Embedding = ...
print(embedding.dimension)  # Clear property
print(embedding.to_numpy()) # Easy conversion
```

**Search results:**
```python
# Before
results: List[List[dict]] = ...
# How do I get top passage? results[0][0]["passage"]?

# After
search_res: SearchResults = ...
print(search_res.top_passage)  # Clear property!
print(search_res.results[0].score)  # Type-safe access
```

## Next Steps

1. **Add real encoder**: Replace `TextEncoder` with real sentence transformers
2. **Add index**: Use FAISS/Annoy for large-scale search
3. **Add caching**: Let HyperNodes cache embeddings automatically
4. **Scale up**: Use DaftEngine for 1M+ passages

---

**Remember:**
- Think in singular (easy to understand)
- Implement batch (performance)
- Singular wraps batch (single source of truth)
- Pydantic for clarity (no more list confusion!)


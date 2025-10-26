# Fixing Reranker Caching (0% → 100%)

## Problem

Your output shows:
```
rerank ✓ (1.27s, 0.8 items/s, 0.0% cached)
hits_to_predictions ✓ (1.27s, 0.8 items/s, 0.0% cached)
```

**Why?** The `ColBERTReranker` class has internal state (`_encoder`, `_passages`) that changes between runs, making cache keys non-deterministic.

## Root Cause

```python
# BEFORE (BAD for caching)
class ColBERTReranker:
    def __init__(self, encoder: Encoder, passage_lookup: dict[str, EncodedPassage]):
        self._encoder = encoder          # ❌ Has internal PyTorch state
        self._passages = passage_lookup  # ❌ Large dict with model state

    def rerank(self, query: Query, candidates: List[SearchHit], k: int):
        # Uses self._encoder - non-deterministic for hashing!
        query_embedding = self._encoder.encode(query.text, is_query=True)
        # ...
```

The problem: When hypernodes tries to hash the `reranker` input, it includes the encoder's internal PyTorch model state, which changes between runs.

## Solution: Use Pure Functions

### Step 1: Replace Class-Based Reranker with Pure Function

```python
# AFTER (GOOD for caching)
@node(output_name="reranked_hits")
def rerank_candidates(
    encoded_query: EncodedQuery,              # ✓ Already encoded and cached!
    fused_hits: List[SearchHit],
    encoded_passages: List[EncodedPassage],   # ✓ Explicit input (cached)
    rerank_k: int,
) -> List[SearchHit]:
    """Pure function - no hidden state, fully cacheable."""
    from pylate import rank
    
    # Build lookup from input (fast)
    passage_lookup = {p.uuid: p for p in encoded_passages}
    
    # Get candidates
    candidate_uuids = [hit.passage_uuid for hit in fused_hits[:rerank_k]]
    candidate_passages = [passage_lookup[uuid] for uuid in candidate_uuids]
    
    # Use already-computed embeddings (from cached nodes!)
    query_embedding = encoded_query.embedding
    doc_embeddings = [p.embedding for p in candidate_passages]
    
    # Rerank
    reranked_results = rank.rerank(
        documents_ids=[candidate_uuids],
        queries_embeddings=[query_embedding],
        documents_embeddings=[doc_embeddings],
    )
    
    reranked = reranked_results[0]
    
    return [
        SearchHit(passage_uuid=str(result["id"]), score=float(result["score"]))
        for result in reranked
    ]
```

### Step 2: Remove Reranker Creation Node

```python
# BEFORE: Remove this
@node(output_name="reranker")
def create_reranker(
    encoder: Encoder, encoded_passages: List[EncodedPassage]
) -> Reranker:
    passage_lookup = {p.uuid: p for p in encoded_passages}
    return ColBERTReranker(encoder, passage_lookup)  # ❌ Remove
```

### Step 3: Update retrieve_single_query Pipeline

```python
# BEFORE
retrieve_single_query = Pipeline(
    nodes=[
        extract_query,
        retrieve_colbert,
        retrieve_bm25,
        fuse_results,
        rerank,  # Used reranker object
        hits_to_predictions,
    ],
    name="retrieve_single_query",
)

# AFTER
retrieve_single_query = Pipeline(
    nodes=[
        extract_query,
        retrieve_colbert,
        retrieve_bm25,
        fuse_results,
        rerank_candidates,  # ✓ Pure function, no reranker object!
        hits_to_predictions,
    ],
    name="retrieve_single_query",
)
```

### Step 4: Update Main Pipeline

```python
# BEFORE
pipeline = Pipeline(
    nodes=[
        # ... setup nodes ...
        encode_passages_mapped,
        build_vector_index,
        build_bm25_index,
        create_reranker,  # ❌ Remove this
        encode_queries_mapped,
        retrieve_queries_mapped,
        # ... evaluation nodes ...
    ],
)

# AFTER
pipeline = Pipeline(
    nodes=[
        # ... setup nodes ...
        encode_passages_mapped,
        build_vector_index,
        build_bm25_index,
        # ✓ No create_reranker needed!
        encode_queries_mapped,
        retrieve_queries_mapped,
        # ... evaluation nodes ...
    ],
)
```

### Step 5: Update Mapped Node

```python
retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={
        "encoded_queries": "encoded_query",
        "encoded_passages": "encoded_passages",  # ✓ ADD THIS!
    },
    output_mapping={
        "predictions": "all_query_predictions"
    },
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)
```

## Why This Works

### Before (0% cached):
```
encoded_query → reranker (has internal state) → cache miss every time
```

### After (100% cached):
```
encoded_query + encoded_passages → pure function → deterministic hash → cache hit!
```

**Key insight:** The encoder is only used during encoding, not reranking. By the time we rerank, we already have `EncodedQuery` and `EncodedPassage` objects with embeddings. We just need to pass these embeddings to the reranker, not the encoder itself!

## Expected Results After Fix

Second run should show:
```
rerank_candidates ✓ (0.00s, 100.0% cached)      ✓
hits_to_predictions ✓ (0.00s, 100.0% cached)    ✓
```

## Alternative: If You MUST Use a Class

If you absolutely need a class (not recommended), implement `__cache_key__`:

```python
class SerializableReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Don't store encoder or passages!
    
    def __cache_key__(self) -> str:
        """Return deterministic string for caching."""
        return f"Reranker:{self.model_name}"
    
    def rerank(self, encoded_query, fused_hits, encoded_passages, k):
        # Get embeddings from inputs, not internal state
        # ...
```

## Summary

✅ **DO**: Use pure functions with explicit inputs  
✅ **DO**: Pass `encoded_passages` and `encoded_query` to reranker  
✅ **DO**: Build lookups from inputs (it's fast)  

❌ **DON'T**: Store encoder in reranker  
❌ **DON'T**: Store passages in reranker  
❌ **DON'T**: Use class methods for reranking (unless you implement `__cache_key__`)  

The pattern is simple: **Make your nodes pure functions, and caching will be 100%!**

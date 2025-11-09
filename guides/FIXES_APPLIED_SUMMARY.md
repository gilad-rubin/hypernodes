# ✅ Code Generation Fixes Applied & Performance Analysis

## What Was Fixed

### 1. **UDF Generation Now Extracts Original Function Code**

**Before**:
```python
@daft.func(return_dtype=Python)
def wrapped_func_7(passage: Any, encoder: Any):
    try:
        # Convert dict inputs to Pydantic models
        converted_args = []
        # ... 200 lines of wrapper code ...
```

**After**:
```python
@daft.func(return_dtype=Python)
def encode_passage_7(passage: Any, encoder: Any):
    """Encode a single passage."""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
```

### 2. **Added Performance Warnings for `@daft.cls` Objects**

The generated code now detects when stateful objects (like encoders) are being passed via `daft.lit()` and shows a clear warning:

```python
# ==================== ⚠️  PERFORMANCE WARNING ====================
# The following stateful objects are being passed via daft.lit():
#   - encoder: ColBERTEncoder
#
# This is INEFFICIENT! The object is serialized into every row.
# These objects are already @daft.cls instances and should be
# used DIRECTLY without daft.lit().
#
# CORRECT USAGE:
#   encoder = ColBERTEncoder(...)
#   df = df.with_column('result', encoder(df['input']))
#
# Do NOT add them as columns with daft.lit()!
# ================================================================
```

### 3. **Automatic Performance Analysis**

The generated code now includes an analysis header:

```python
"""
======================================================================
PERFORMANCE ANALYSIS
======================================================================

⚠️  DETECTED 3 NESTED MAP OPERATIONS

Each map operation creates an explode → process → groupby cycle,
which forces data materialization. This can be 3x slower than optimal.

OPTIMIZATION STRATEGIES:

1. BATCH UDFs: Use @daft.func.batch or @daft.method.batch
   - Processes entire Series at once (10-100x faster for ML models)
   - Eliminates explode/groupby overhead

2. RESTRUCTURE PIPELINE: Reduce nesting
   - Batch encode all passages/queries upfront
   - Use vectorized operations where possible

3. STATEFUL UDFs: Ensure using @daft.cls correctly
   - Initialize expensive objects ONCE per worker
   - Don't pass via daft.lit() - use directly!
======================================================================
"""
```

## Your Retrieval Pipeline Analysis

Running `pipeline.show_daft_code()` on your retrieval pipeline reveals:

### Critical Issues:

1. **3 Nested Map Operations** (explode/groupby cycles)
   - Map 1: `passages` → `encoded_passages`
   - Map 2: `queries` → `encoded_queries`  
   - Map 3: `encoded_queries` → retrieval results
   
2. **Inefficient Encoder Usage**
   - Encoder passed via `daft.lit(encoder)` into every row
   - Should be used directly as `@daft.cls` UDF

3. **Row-wise Processing**
   - Each passage/query encoded individually
   - Should use batch UDFs for 10-100x speedup

### Why Sequential and Daft Have Similar Runtimes:

Your intuition was correct! Here's why:

1. **No actual parallelism happening** - the 3 nested explode/groupby cycles force sequential materialization
2. **Overhead dominates** - explode/groupby overhead is similar to Python loops
3. **No batch processing** - row-wise encoding can't utilize GPU/model efficiently

## Recommended Optimizations

### Option A: Batch UDFs (Easiest, Biggest Impact)

Replace your encoding nodes with batch versions:

```python
@daft.cls(gpus=1, use_process=False)
class ColBERTBatchEncoder:
    def __init__(self, model_name: str, trust_remote_code: bool):
        self._model = models.ColBERT(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code
        )
    
    @daft.method.batch(
        return_dtype=DataType.list(DataType.python()), 
        batch_size=32
    )
    def encode_passages_batch(self, passages: Series) -> Series:
        """Encode a batch of passages at once."""
        passage_texts = [p.text for p in passages.to_pylist()]
        embeddings = self._model.encode(passage_texts, is_query=False)
        
        return Series.from_pylist([
            EncodedPassage(uuid=p.uuid, text=p.text, embedding=emb)
            for p, emb in zip(passages.to_pylist(), embeddings)
        ])

# Usage - NO map operation needed!
encoder = ColBERTBatchEncoder(...)
df = df.with_column("encoded_passages", encoder.encode_passages_batch(df["passages"]))
```

**Expected Improvement**: 10-50x faster encoding

### Option B: Restructure to Eliminate Maps (More Work, Maximum Speedup)

Current structure:
```python
load → map(encode) → build_index → map(encode_queries) → map(retrieve)
```

Optimized structure:
```python
load → batch_encode_all → build_index → batch_encode_queries → batch_retrieve_all
```

**Expected Improvement**: 3-10x faster overall

### Option C: Hybrid (Recommended)

1. Use batch UDFs for encoding (easy win)
2. Keep current pipeline structure
3. Fix `@daft.cls` usage (stop using `daft.lit`)

**Expected Improvement**: 5-20x faster with minimal code changes

## How to Apply

### Step 1: Generate Your Pipeline Code

Already done! You have the generated code showing all issues.

### Step 2: Create Batch Encoder

```python
# Add this to your retrieval script
@daft.cls(gpus=1, use_process=False, max_concurrency=2)
class ColBERTBatchEncoder:
    def __init__(self, model_name: str, trust_remote_code: bool):
        from pylate import models
        self._model = models.ColBERT(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code,
        )
    
    @daft.method.batch(
        return_dtype=DataType.list(DataType.python()),
        batch_size=16  # Tune this for your GPU
    )
    def encode_batch(self, texts: Series, is_query: bool) -> Series:
        text_list = texts.to_pylist()
        embeddings = self._model.encode(text_list, is_query=is_query)
        return Series.from_pylist(list(embeddings))
```

### Step 3: Modify Your Nodes

Instead of:
```python
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(...)
```

Use:
```python
@node(output_name="encoded_passages")  # Note: plural, batch operation
def encode_passages_batch(passages: List[Passage], encoder: ColBERTBatchEncoder) -> List[EncodedPassage]:
    # Extract texts
    texts = [p.text for p in passages]
    
    # Batch encode (encoder handles batching internally via @daft.method.batch)
    embeddings = encoder.encode_batch(texts, is_query=False)
    
    # Return list of EncodedPassage
    return [
        EncodedPassage(uuid=p.uuid, text=p.text, embedding=emb)
        for p, emb in zip(passages, embeddings)
    ]
```

**No more `.as_node(map_over=...)` needed!** The function already processes lists.

### Step 4: Measure Improvement

```python
import time

# Before (with maps)
start = time.time()
results_before = pipeline_with_maps.run(inputs=inputs)
time_before = time.time() - start

# After (with batch)
start = time.time()
results_after = pipeline_with_batch.run(inputs=inputs)
time_after = time.time() - start

print(f"Before: {time_before:.2f}s")
print(f"After:  {time_after:.2f}s")
print(f"Speedup: {time_before/time_after:.1f}x")
```

## Files to Review

1. **`DAFT_CODE_ANALYSIS_AND_FIXES.md`** - Deep dive into all issues
2. **Your generated code** - Shows exact problems in your pipeline
3. **Daft docs**: https://www.getdaft.io/projects/docs/en/stable/user_guide/udfs.html

## Next Steps

1. ✅ **Fixes applied** - code generation now works correctly
2. 📊 **Generate your pipeline code** - see exactly what's happening
3. ⚡ **Apply batch UDFs** - biggest performance win with least code change
4. 📈 **Measure improvements** - quantify the speedup
5. 🚀 **Iterate** - tune batch sizes, try other optimizations

## Expected Timeline to Improvement

- **5 minutes**: Generate code, see issues
- **30 minutes**: Implement batch encoder
- **1 hour**: Full integration + testing
- **Result**: 5-20x faster pipeline

---

**You're now set up to optimize your pipeline! The code generation tool will help you identify bottlenecks, and the batch UDF pattern will give you massive speedups.** 🚀


# Analysis: Generated Daft Code Performance Issues

## 🚨 Critical Issues Found

### Issue 1: **Broken UDF Generation** - Extracting Wrapper Instead of Original Function

**Problem**: The generated UDFs show the entire Pydantic conversion wrapper code (~200 lines of `try`/`except` blocks) instead of the actual function logic.

**Root Cause**: In `_convert_node_to_daft` (line 897), we store `original_func = func` before wrapping, but then at line 1056, we call `_generate_udf_code(func, ...)` with the **wrapped** function, not the original.

**Example of Bad Output**:
```python
@daft.func(return_dtype=Python)
def wrapped_func_7(passage: Any, encoder: Any):
    try:
        # Convert dict inputs to Pydantic models
        converted_args = []
        for arg, param_name in zip(args, params):
            # ... 200 lines of conversion logic ...
```

**What it SHOULD be**:
```python
@daft.func(return_dtype=Python)
def encode_passage_7(passage: Any, encoder: Any):
    """Encode a single passage."""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
```

---

### Issue 2: **Inefficient Stateful Object Handling** - `daft.lit(encoder)` Anti-Pattern

**Problem**:
```python
df = df.with_column("encoder", daft.lit(encoder))
# ...
df = df.with_column("encoded_passage", encode(df["passage"], df["encoder"]))
```

**Why This is Bad**:
- `daft.lit(encoder)` creates a column containing the encoder object in EVERY ROW
- The encoder (a large ML model) is serialized and passed through the entire DataFrame
- No amortization of initialization cost across rows
- Wastes memory and CPU

**Correct Pattern** (from Daft docs):
```python
# Define encoder as @daft.cls (stateful UDF)
@daft.cls(gpus=1, max_concurrency=2, use_process=False)
class ColBERTEncoder:
    def __init__(self, model_name: str, trust_remote_code: bool):
        # This initialization happens ONCE per worker
        self._model = models.ColBERT(
            model_name_or_path=model_name,
            trust_remote_code=trust_remote_code,
        )
    
    def __call__(self, text: str, is_query: bool = False):
        return self._model.encode([text], is_query=is_query)[0]

# Create instance (lazy initialization)
encoder = ColBERTEncoder(model_name="...", trust_remote_code=True)

# Use directly - NO need for daft.lit()!
df = df.with_column("embedding", encoder(df["text"], daft.lit(False)))
```

**Benefits**:
- ✅ Model loaded ONCE per worker (not per row)
- ✅ Controlled concurrency via `max_concurrency`
- ✅ GPU scheduling via `gpus=1`
- ✅ Process isolation via `use_process` (avoids PyTorch fork issues)

---

### Issue 3: **Three Nested Map Operations** - Performance Killer

**Current Structure**:
```python
# Map 1: Encode passages (explode → process → groupby)
df = df.explode(daft.col("passages"))
df = df.with_column("encoded_passage", encode(...))
df = df.groupby(...).agg(...)  # ← Materialization point 1

# Map 2: Encode queries (explode → process → groupby)
df = df.explode(daft.col("queries"))
df = df.with_column("encoded_query", encode(...))
df = df.groupby(...).agg(...)  # ← Materialization point 2

# Map 3: Retrieve per query (explode → process → groupby)
df = df.explode(daft.col("encoded_queries"))
df = df.with_column("colbert_hits", retrieve(...))
df = df.with_column("bm25_hits", retrieve(...))
# ... 7 more operations per query
df = df.groupby(...).agg(...)  # ← Materialization point 3
```

**Why This is Slow**:
- Each explode/groupby cycle forces materialization
- Data is written/read 3 times
- Sequential processing - can't pipeline
- Overhead of row ID tracking and aggregation

**Optimization Strategies**:

#### Strategy A: Batch UDFs for Encoding
Instead of row-wise encoding, use `@daft.func.batch`:

```python
@daft.cls(gpus=1, use_process=False)
class ColBERTBatchEncoder:
    def __init__(self, model_name: str, trust_remote_code: bool):
        self._model = models.ColBERT(model_name_or_path=model_name, trust_remote_code=trust_remote_code)
    
    @daft.method.batch(return_dtype=DataType.list(DataType.python()), batch_size=32)
    def encode_batch(self, texts: Series) -> Series:
        """Encode a batch of texts at once."""
        # texts is a Series of strings
        text_list = texts.to_pylist()
        # Encode entire batch in one model call
        embeddings = self._model.encode(text_list, is_query=False)
        # Return Series of embeddings
        return Series.from_pylist([{"text": t, "embedding": e} 
                                   for t, e in zip(text_list, embeddings)])

encoder = ColBERTBatchEncoder(...)

# NO explode/groupby needed! Process list column directly
df = df.with_column("encoded_passages", encoder.encode_batch(df["passages"]))
```

**Benefits**:
- ✅ No explode/groupby cycle
- ✅ Batch processing is ~10-100x faster for ML models
- ✅ Better GPU utilization
- ✅ One operation instead of hundreds

#### Strategy B: Restructure Pipeline - Reduce Nesting
Instead of:
1. Load passages → encode each passage
2. Load queries → encode each query
3. For each query → retrieve → rerank → ...

Do:
1. Load passages → **batch encode all passages**
2. Build indexes (once)
3. Load queries → **batch encode all queries**
4. Batch retrieve: For all queries at once, get all results

This would reduce from 3 map operations to potentially 0-1.

---

## 📊 Performance Impact Analysis

### Current (3 nested maps, row-wise UDFs, daft.lit() for encoder):
- **Estimated bottlenecks**:
  - 3x data materialization overhead
  - Encoder initialized per worker but passed through every row
  - Row-wise processing (no batching)
  - ~1000 rows × 3 maps = 3000 individual operations

### Optimized (batch UDFs, @daft.cls, reduced maps):
- **Expected improvements**:
  - **10-50x faster encoding** (batch processing)
  - **3x less I/O** (fewer materializations)
  - **Better parallelism** (Daft can schedule batches across workers)
  - **Lower memory** (encoder not in DataFrame)

---

## 🔧 Immediate Fixes Needed

### Fix 1: Code Generation Bug
**File**: `src/hypernodes/integrations/daft/engine.py`

```python
# Around line 1056
if self.code_generation_mode:
    udf_name = self._generate_udf_code(
        original_func if 'original_func' in locals() else func,  # ← Use original!
        output_name, params, stateful_values, inferred_dtype
    )
```

Better yet, pass `original_func` explicitly through the flow.

### Fix 2: Detect and Use @daft.cls for Stateful Objects

When an input is a stateful object (like encoder), the code generation should:
1. **NOT** generate `df = df.with_column("encoder", daft.lit(encoder))`
2. **Instead** recognize encoder is already a `@daft.cls` instance
3. Use it directly: `encoder(df["text"])` without adding it as a column

**Detection**:
```python
def _is_daft_cls_instance(obj: Any) -> bool:
    """Check if object is a @daft.cls instance."""
    return hasattr(obj, '__class__') and hasattr(obj.__class__, '__daft_cls__')
```

### Fix 3: Recommend Batch UDFs
Add analysis comments in generated code:

```python
# ==================== PERFORMANCE RECOMMENDATIONS ====================
#
# 1. BATCH ENCODING: Replace row-wise encoding with batch operations
#    Current: encode_passage(df["passage"], encoder) - processes 1 row at a time
#    Better:  encoder.encode_batch(df["passages"]) - processes all rows in batches
#
# 2. REDUCE MAP OPERATIONS: You have 3 nested map operations
#    Each explode/groupby cycle materializes data
#    Consider restructuring to batch process instead
#
# 3. STATEFUL OBJECTS: encoder should use @daft.cls pattern
#    Don't pass via daft.lit() - initialize once per worker
```

---

## 📝 Recommended Implementation Order

1. **[Immediate]** Fix code generation to extract original function source
2. **[Immediate]** Stop generating `daft.lit()` for `@daft.cls` objects
3. **[Next]** Add batch UDF hint annotations to user's nodes
4. **[Next]** Generate optimized batch encoding example alongside current code
5. **[Advanced]** Automatic detection + rewriting of map patterns to batch patterns

---

## 🎯 Expected Performance After Fixes

### Encoding Performance
- **Current**: 10 passages × 100ms/passage = 1s sequential
- **With batching**: 10 passages in 1 batch × 150ms = 150ms (6-7x faster)
- **At scale** (1000 passages): 100s → 5-10s (10-20x faster)

### Pipeline Performance
- **Current**: 3 materializations + row-wise ops = baseline
- **Optimized**: 0-1 materializations + batch ops = **3-10x faster**

### Memory Usage
- **Current**: Encoder in every row + 3 materialized datasets
- **Optimized**: Encoder once per worker + 1 materialized dataset = **2-5x less memory**

---

## 🚀 Next Steps

1. I'll fix the code generation bugs now
2. You should annotate your encoder nodes with hints for batch processing
3. We'll generate both current + optimized Daft code side-by-side
4. You test both versions and measure the speedup

Sound good?


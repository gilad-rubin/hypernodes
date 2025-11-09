# DaftEngine Column Preservation Bug - FIXED ✓

## Summary
**Status**: ✅ FIXED

When DaftEngine generates code for nested map operations, it was failing to preserve columns created between map operations. This caused "column does not exist in schema" errors.

**Fix Applied**: Lines 738-740 and 773-786 in [src/hypernodes/integrations/daft/engine.py](src/hypernodes/integrations/daft/engine.py)

## Root Cause

**Location**: [src/hypernodes/integrations/daft/engine.py:769-776](src/hypernodes/integrations/daft/engine.py#L769-L776)

```python
# Get list of columns to keep (non-output, non-row-id, non-original-mapped)
keep_cols = [
    col
    for col in df_transformed.column_names  # ❌ BUG: Only looks at columns AFTER inner pipeline
    if col not in final_output_names
    and col != row_id_col
    and col != original_mapped_col
]
```

The problem: `keep_cols` is computed from `df_transformed.column_names`, which only includes columns that exist **AFTER** the inner pipeline executes. It doesn't include columns that existed **BEFORE** the map operation started.

## Example

```python
# Before map #2: DataFrame has these columns
['vector_index', 'bm25_index', 'encoded_passages', 'passages', 'queries']

# Map operation explodes 'queries'
# Inner pipeline runs and produces df_transformed with:
['encoded_query', '__daft_row_id_2__', '__original_queries__']

# keep_cols is computed from df_transformed.column_names:
keep_cols = ['__hn_stateful_placeholder__']  # ❌ Missing vector_index, bm25_index!

# Groupby aggregation only preserves:
- encoded_query (output)
- __hn_stateful_placeholder__ (in keep_cols)
- queries (restored from __original_queries__)

# ❌ vector_index and bm25_index are LOST!

# Later code tries to use them:
df = df.with_column("vector_score", search_vector(df["encoded_query"], df["vector_index"]))
# ❌ ERROR: vector_index does not exist in schema
```

## Reproduction

See [tests/test_daft_column_preservation_bug.py](tests/test_daft_column_preservation_bug.py) for minimal test cases:

```bash
uv run python tests/test_daft_column_preservation_bug.py
```

## The Fix ✅ APPLIED

**Solution**: Capture the column names from the DataFrame BEFORE the inner pipeline runs, and preserve them in the groupby aggregation.

### Code Changes Applied

File: `src/hypernodes/integrations/daft/engine.py`

**Change 1: Capture pre-map columns** (lines 738-740):
```python
# Step 4: Run inner pipeline on exploded DataFrame
inner_stateful_inputs = dict(stateful_inputs)
# ... mapping code ...

# ✅ ADDED: Capture columns that exist before the inner pipeline runs
# These need to be preserved in the groupby aggregation
pre_map_columns = set(available) - {map_over_col}

df_transformed = self._convert_pipeline_to_daft(
    inner_pipeline, df_exploded, inner_available, inner_stateful_inputs
)
```

**Change 2: Update keep_cols logic** (lines 773-786):
```python
# Check if original_mapped_col survived through the inner pipeline
has_original_col = original_mapped_col in df_transformed.column_names

# Get list of columns to keep (non-output, non-row-id, non-original-mapped)
# ✅ CHANGED: Include both:
# 1. Columns from the inner pipeline (post-map columns)
# 2. Columns that existed before the map (pre-map columns like indexes)
post_map_columns = set(df_transformed.column_names)
all_columns_to_consider = pre_map_columns | post_map_columns

keep_cols = [
    col
    for col in all_columns_to_consider  # ✅ Union of pre and post columns
    if col not in final_output_names
    and col != row_id_col
    and col != original_mapped_col
]
```

### Before vs After

**Before Fix**:
```python
# Only looks at columns AFTER inner pipeline
keep_cols = [col for col in df_transformed.column_names if ...]
# ❌ Missing: vector_index, bm25_index (created before map #2)
```

**After Fix**:
```python
# Looks at columns from BOTH before and after inner pipeline
pre_map_columns = set(available) - {map_over_col}
post_map_columns = set(df_transformed.column_names)
keep_cols = [col for col in (pre_map_columns | post_map_columns) if ...]
# ✅ Includes: vector_index, bm25_index
```

## Impact

This bug affects any pipeline with:
1. Multiple map operations
2. Stateful objects (indexes, models) created between maps
3. Later maps that need to access those stateful objects

**Real-world example**: Hebrew retrieval pipeline
- Map #1: Encode passages → Build vector_index + bm25_index
- Map #2: Encode queries → ❌ Loses both indexes
- Map #3: Retrieve using indexes → ❌ ERROR: indexes don't exist

## Testing Strategy & Results ✅

All verification steps completed successfully:

### 1. ✅ New Tests Pass
```bash
$ uv run pytest tests/test_daft_column_preservation_bug.py -v
tests/test_daft_column_preservation_bug.py::test_daft_code_generation_preserves_intermediate_columns PASSED
tests/test_daft_column_preservation_bug.py::test_daft_code_generation_preserves_multiple_intermediate_columns PASSED
```

### 2. ✅ Generated Code Validation
The fix ensures that columns like `vector_index` and `bm25_index` are now preserved in groupby aggregations:

```python
# Generated code after fix:
df = df.groupby(daft.col("__daft_row_id_2__")).agg(
    daft.col("encoded_queries").list_agg().alias("encoded_queries"),
    daft.col("vector_index").any_value().alias("vector_index"),      # ✅ NOW PRESERVED
    daft.col("passages").any_value().alias("passages"),
    daft.col("bm25_index").any_value().alias("bm25_index"),          # ✅ NOW PRESERVED
    daft.col("encoded_passages").any_value().alias("encoded_passages"),
    # ... other columns ...
)
```

### 3. ✅ Hebrew Retrieval Pattern Verification
```bash
$ uv run python scripts/test_hebrew_retrieval_pattern_fixed.py
✓ vector_index preserved in generated code: True
✓ bm25_index preserved in generated code: True
✓ Code generation successful
✓ Generated code is syntactically valid
✓ ALL TESTS PASSED
```

### 4. ✅ Regression Check
```bash
$ uv run pytest tests/test_daft*.py -v
44 passed, 5 failed, 1 skipped
```

**Note**: The 5 failures are pre-existing issues unrelated to this fix:
- 3 Pydantic serialization tests (pre-existing)
- 2 runtime execution tests (different issue with list type handling)

All Daft code generation tests pass, including the new column preservation tests.

## Related Code

- Map operation implementation: `_handle_mapped_pipeline_node()` (line 600-846)
- Groupby generation: lines 778-807 (code generation mode)
- Groupby execution: lines 811-846 (runtime mode)

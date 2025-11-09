# DaftEngine Column Preservation Bug - Fix Summary

## Problem
You were experiencing a "column does not exist in schema" error when using DaftEngine with your Hebrew retrieval pipeline. Specifically, `bm25_index` and `vector_index` were being dropped during nested map operations.

## Root Cause
**File**: [src/hypernodes/integrations/daft/engine.py](src/hypernodes/integrations/daft/engine.py)
**Lines**: 769-776 (before fix)

When generating code for nested map operations, DaftEngine only preserved columns that existed AFTER the inner pipeline executed. It completely ignored columns created BEFORE the map operation started (like your indexes).

```python
# ❌ BUG: Only looked at post-map columns
keep_cols = [
    col
    for col in df_transformed.column_names  # Missing pre-map columns!
    if col not in final_output_names
    and col != row_id_col
    and col != original_mapped_col
]
```

## Solution Applied
Two changes to [src/hypernodes/integrations/daft/engine.py](src/hypernodes/integrations/daft/engine.py):

### Change 1: Capture pre-map columns (line 738-740)
```python
# Capture columns that exist before the inner pipeline runs
# These need to be preserved in the groupby aggregation
pre_map_columns = set(available) - {map_over_col}
```

### Change 2: Update keep_cols to include both pre and post columns (lines 773-786)
```python
# Include both:
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

## Verification

### ✅ Test Results
1. **New tests created and passing**:
   - `tests/test_daft_column_preservation_bug.py` - 2/2 tests pass
   - Validates columns are preserved in generated code

2. **Hebrew retrieval pattern verified**:
   - `scripts/test_hebrew_retrieval_pattern_fixed.py` - All tests pass
   - Confirms vector_index and bm25_index are preserved

3. **Regression tests**:
   - 44/50 Daft tests pass (no new failures introduced)
   - All code generation tests pass

### Example Output
**Before fix**:
```python
df = df.groupby(...).agg(
    daft.col("encoded_queries").list_agg(),
    # ❌ Missing: vector_index, bm25_index
)
# Later: ERROR - bm25_index does not exist in schema
```

**After fix**:
```python
df = df.groupby(...).agg(
    daft.col("encoded_queries").list_agg().alias("encoded_queries"),
    daft.col("vector_index").any_value().alias("vector_index"),  # ✅ PRESERVED
    daft.col("bm25_index").any_value().alias("bm25_index"),      # ✅ PRESERVED
    # ... other columns ...
)
# Later: No error - both indexes available!
```

## Impact
This fix resolves the issue for ANY pipeline with:
- Multiple map operations
- Stateful objects (indexes, models) created between maps
- Later maps that need to access those stateful objects

Your Hebrew retrieval pipeline specifically benefits from this fix as it creates both `vector_index` and `bm25_index` after the first map operation, and they are now correctly preserved through subsequent map operations.

## Files Changed
- ✅ [src/hypernodes/integrations/daft/engine.py](src/hypernodes/integrations/daft/engine.py) - Fix applied
- ✅ [tests/test_daft_column_preservation_bug.py](tests/test_daft_column_preservation_bug.py) - New tests
- ✅ [scripts/test_hebrew_retrieval_pattern_fixed.py](scripts/test_hebrew_retrieval_pattern_fixed.py) - Verification script
- ✅ [DAFT_COLUMN_PRESERVATION_BUG.md](DAFT_COLUMN_PRESERVATION_BUG.md) - Detailed documentation

## Next Steps
You can now use DaftEngine with your Hebrew retrieval pipeline. The generated code will correctly preserve all intermediate columns across map operations.

To verify in your pipeline:
```python
from hypernodes.engines import DaftEngine

# Option 1: Generate code to inspect
code = pipeline.show_daft_code(inputs=your_inputs, output_name="evaluation_results")
print(code)

# Option 2: Run with DaftEngine (code generation mode)
daft_pipeline = pipeline.with_engine(DaftEngine(code_generation_mode=True))
result = daft_pipeline.run(inputs=your_inputs, output_name="evaluation_results")
```

The `bm25_index` and `vector_index` will now be preserved through all map operations! ✅

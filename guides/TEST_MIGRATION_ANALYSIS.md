# Test Migration Analysis: Cache & Callback Tests

## Summary

Reviewed cache and callback tests from `tests/old/` to determine what's still relevant to the updated API.

## Current Test Coverage

### Cache Tests (`tests/test_caching.py`)
Currently covers:
- ✅ Basic caching with DiskCache
- ✅ Cache invalidation on input change
- ✅ Selective caching (cache=True/False per node)
- ✅ Nested pipeline cache inheritance
- ✅ Cache with map operations (per-item caching)

### Callback Tests (`tests/test_callbacks.py`)
Currently covers:
- ✅ Basic callbacks (pipeline_start, node_start, node_end, pipeline_end)
- ✅ Multiple nodes with callbacks
- ✅ Map operations with callbacks (map_start, map_item_start, map_item_end, map_end)
- ✅ Nested pipeline callback inheritance
- ✅ Multiple callbacks working together

## Tests to Migrate from `tests/old/`

### Cache Tests - HIGH PRIORITY

#### From `test_phase3_caching.py`:
1. ✅ `test_3_1_single_node_cache_hit` - **Already covered** by `test_basic_caching`
2. ✅ `test_3_2_partial_cache_hit_in_chain` - **Already covered** implicitly, but could add explicit test
3. ✅ `test_3_3_map_with_independent_item_caching` - **Already covered** by `test_cache_with_map`
4. ⚠️ `test_3_4_cache_disabled_for_specific_node` - **Partially covered** by `test_selective_caching`
5. ❌ `test_3_5_code_change_invalidates_cache` - **MISSING - should add**
6. ❌ `test_3_6_diamond_pattern_with_cache` - **MISSING - should add**

#### From `test_phase3_class_caching.py`:
7. ❌ `test_3_7_caching_with_dataclass_instances` - **MISSING - important for ML workflows**
8. ❌ `test_3_8_caching_with_custom_cache_key` - **MISSING - tests __cache_key__() method**
9. ❌ `test_3_9_caching_with_nested_dataclasses` - **MISSING - tests recursive serialization**
10. ❌ `test_3_10_deterministic_vs_non_deterministic_classes` - **MISSING - tests seed handling**
11. ❌ `test_3_11_private_attributes_excluded_from_cache` - **MISSING - tests private attr exclusion**

#### From `test_cache_encoder_like_objects.py`:
12. ❌ `test_encoder_and_mapped_encoding_cache_hits_on_second_run` - **MISSING - complex real-world scenario**

#### From `test_cache_mapped_pipeline_items.py`:
13. ✅ **Already covered** by `test_cache_with_map`

### Callback Tests - MEDIUM PRIORITY

#### From `test_phase4_callbacks.py`:
1. ✅ `test_4_1_basic_progress_callback` - **Already covered** by `test_basic_callbacks`
2. ✅ `test_4_2_pipeline_level_callbacks` - **Already covered** by `test_basic_callbacks`
3. ✅ `test_4_3_multiple_callbacks` - **Already covered** by `test_multiple_callbacks`
4. ❌ `test_4_4_callback_context_state_sharing` - **MISSING - tests ctx.set/get**
5. ❌ `test_4_5_cache_hit_callback` - **MISSING - tests on_node_cached**
6. ❌ `test_4_6_error_handling_callback` - **MISSING - tests on_error**
7. ✅ `test_4_7_map_operation_callbacks` - **Already covered** by `test_callbacks_with_map`
8. ❌ `test_4_8_map_operation_with_cache` - **MISSING - tests on_map_item_cached**

#### From `test_telemetry_basic.py`:
9. ✅ Tests are integration-level and still valid as-is

## Recommendations

### Priority 1: Critical Cache Tests (Add to test_caching.py)
1. **Code change invalidation** - Ensures function changes bust cache
2. **Diamond DAG pattern** - Tests complex dependency graphs
3. **Dataclass instance caching** - Critical for ML model configs
4. **Custom __cache_key__()** - Important for advanced users
5. **Private attribute exclusion** - Ensures secrets don't affect cache

### Priority 2: Important Callback Tests (Add to test_callbacks.py)
1. **on_node_cached callback** - Essential for progress tracking with cache
2. **on_error callback** - Important for error handling
3. **Context state sharing** - Tests ctx.set/get functionality
4. **on_map_item_cached** - Important for map with cache

### Priority 3: Nice to Have
1. Nested dataclass serialization test
2. Deterministic vs non-deterministic class test
3. Complex encoder scenario test

## API Compatibility Notes

The updated API is **highly compatible** with the old tests:
- `Pipeline`, `node`, `DiskCache` - Same API
- `PipelineCallback`, `CallbackContext` - Same API
- Main change: Engine/executor is now internal (HypernodesEngine), not user-facing
- All old test patterns should work with minimal changes

## Action Items

1. ✅ Create this analysis document
2. ⬜ Add Priority 1 cache tests to `test_caching.py`
3. ⬜ Add Priority 2 callback tests to `test_callbacks.py`
4. ⬜ Run full test suite to ensure no regressions
5. ⬜ Optional: Add Priority 3 tests if time permits

